from torch.utils.data import Dataset
import os
import requests
import numpy as np
import torch
import pickle


class TextDataSet(Dataset):
    def __init__(self, split : str = "train", block_size : int = 1024) -> None:
        self.split = split
        assert self.split in ["train", "val"]

        self.block_size = block_size

        bin_file_name = os.path.join(os.path.dirname(__file__), f'data/{self.split}.npy')
        if not os.path.exists(bin_file_name):
            self.save_to_bin()

        self.ids = np.load(bin_file_name)
    

    def save_to_bin(self):
        input_file_path = os.path.join(os.path.dirname(__file__), 'data/input.txt')
        with open(input_file_path, 'r') as f:
            data = f.read()
        print(f"length of dataset in characters: {len(data):,}")

        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        print("all the unique characters:", ''.join(chars))
        print(f"vocab size: {vocab_size:,}")

        stoi = { ch : i for i,ch in enumerate(chars) }
        itos = { i : ch for i,ch in enumerate(chars) }

        def encode(s):
            return [stoi[c] for c in s] # encoder: take a string, output a list of integers
        def decode(l):
            return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

        n = len(data)
        if self.split == "train":
            data = data[:int(n*0.9)]
        else:
            data = data[int(n*0.9):]

        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
        }
        with open(os.path.join(os.path.dirname(__file__), 'data/meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)


        n = len(data)

        ids = encode(data)
        print(f"{self.split} has {len(ids):,} tokens")

        ids = np.array(ids, dtype=np.uint16)
        bin_file_name = os.path.join(os.path.dirname(__file__), f'data/{self.split}.npy')
        np.save(bin_file_name, ids)

    def __len__(self):
        return self.ids.shape[0] - self.block_size

    def __getitem__(self, idx):
        return torch.from_numpy(self.ids[idx:idx + self.block_size]), torch.from_numpy(self.ids[idx+1:idx +self.block_size+1])


class Vocab:
    def __init__(self):
        meta_path = os.path.join(os.path.join(os.path.dirname(__file__), 'data/meta.pkl'))
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            meta_vocab_size = meta['vocab_size']
            print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
        self.stoi = meta['stoi']
        self.itos = meta['itos']
        self.vocab_size = meta['vocab_size']

    def encode(self, s):
        return [self.stoi[c] for c in s] # encoder: take a string, output a list of integers
    
    def decode(self, l):
        return ''.join([self.itos[i] for i in l]) # decoder: take a list of integers, output a string

if __name__ == "__main__":
    dataset = TextDataSet("train")
    x, y = dataset[0] 
    vocab = Vocab()
    s = vocab.decode(x.numpy())
    # print("x: ", s)
    n = vocab.decode(y.numpy())
    # print("n", n)

    x, y = dataset[len(dataset) - 1] 
    print(y)
    s = vocab.decode(x.numpy())
    print("xxxxxx: \n", s)
    n = vocab.decode(y.numpy())
    print("nnnnnn: \n", n)

    

