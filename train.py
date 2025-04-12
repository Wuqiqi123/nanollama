
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import math
from tqdm import tqdm

from torch.utils.data import DataLoader

from gpt2 import GPTConfig, GPT

from dataset import TextDataSet, Vocab

import numpy as np

device = 'cuda'

decay_lr = True # whether to decay the learning rate
out_dir = 'out'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 1 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
always_save_checkpoint = True # if True, always save a checkpoint after each eval

# 'float32''bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 

ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

print("ddp", ddp)

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size

if master_process:
    os.makedirs(out_dir, exist_ok=True)


torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

dataset = 'openwebtext'
data_dir = os.path.join('data', dataset)


batch_size = 64

train_model_args = {
    "n_layer": 6,
    "n_head": 6, 
    "n_embd": 384,
    "dropout": 0.2,
    "bias": False,
    "block_size": 256,

}

train_dataloader = DataLoader(dataset=TextDataSet("train", train_model_args["block_size"]), batch_size = batch_size, shuffle=True)
val_dataloader = DataLoader(dataset=TextDataSet("val", train_model_args["block_size"]), batch_size = batch_size, shuffle=True)

vocab = Vocab()

print(f"vocab size: {vocab.vocab_size}")
vocab.vocab_size

iter_num = 0
best_val_loss = 1e9


print(f"train_model_args: {train_model_args}")

gptconf = GPTConfig(**train_model_args)

eval_interval = 2000
weight_decay = 1e-1
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta1 = 0.9
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
warmup_iters = 100 # not super necessary potentially


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

running_mfu = -1.0

def train():
    model = GPT(gptconf)

    model.to(device)

    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    model = torch.compile(model)

    if ddp:
        raw_model = model
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # logging
    if master_process:
        import wandb
        wandb.init(project="owt", name="nanogpt", config=train_model_args)

    iter_num = 0
    for epoch in range(10):
        for batch_idx, (X, Y) in tqdm(enumerate(train_dataloader)):
            with torch.amp.autocast(device_type=device_type, dtype=ptdtype):
                X = X.to(device)
                Y = Y.to(device)
                logits, loss = model(X, Y)

            # determine and set the learning rate for this iteration
            lr = get_lr(batch_idx) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            

            # if iter_num % eval_interval == 0 and master_process:
            #     losses = estimate_loss()
            #     print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            #     wandb.log({
            #             "iter": iter_num,
            #             "train/loss": losses['train'],
            #             "val/loss": losses['val'],
            #             "lr": lr,
            #             "mfu": running_mfu*100, # convert to percentage
            #         })

            #     if losses['val'] < best_val_loss or always_save_checkpoint:
            #         best_val_loss = losses['val']
            #         if iter_num > 0:
            #             checkpoint = {
            #                 'model': raw_model.state_dict(),
            #                 'optimizer': optimizer.state_dict(),
            #                 'train_model_args': train_model_args,
            #                 'iter_num': iter_num,
            #                 'best_val_loss': best_val_loss,
            #             }
            #             print(f"saving checkpoint to {out_dir}")
            #             torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if ddp:
        destroy_process_group()
        

if __name__ == "__main__":
    train()
                    

