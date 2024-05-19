from Tokenizer import BasicTextTokenizer

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# Class loading mini tiny_shakespeare dataset encode/tokenize it and create dataloaders
class Data():
    def __init__(self, ration = 0.9, bite_pair_encoding=0):
        print("---Data loading---")
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            text = text[:int(len(text)*0.1)]
            
        self.tokenizer = BasicTextTokenizer(text, bite_pair_encoding)
        
        print("---Data Encoding---")
        self.data = self.tokenizer.encode(text)
        n = int(len(self.data)*ration)
        self.train_data = self.data[:n]
        self.test_data = self.data[n:]

    #return train, validation and test dataloader
    #ratioTT is the ratio btw the size of train+validation and test
    #ratioTT is the ratio btw the size of train and validation
    #cut the ratio of a partial use of the dataset
    def getDataLoaders(self, block_size, batch_size, ratioTT = 0.9, ratioTV = 0.9, cut=1):
        print("---Creating DataLoaders---")
        
        data_test = self.data[int(len(self.data)*ratioTT):]
        data_train = self.data[:int(len(self.data)*ratioTT)]

        data_valid = self.data[int(len(data_train)*ratioTV):]
        data_train = self.data[:int(len(data_train)*ratioTV)]

        data_train_block = [(data_train[i:i+block_size], data_train[i+1:i+block_size+1]) for i in tqdm(range(int(len(data_train)*cut)-block_size-1))]
        data_valid_block = [(data_valid[i:i+block_size], data_valid[i+1:i+block_size+1]) for i in tqdm(range(int(len(data_valid)*cut)-block_size-1))]
        data_test_block = [(data_test[i:i+block_size], data_test[i+1:i+block_size+1]) for i in tqdm(range(int(len(data_test)*cut)-block_size-1))]

        return DataLoader(data_train_block, batch_size=batch_size), DataLoader(data_valid_block, batch_size=batch_size), DataLoader(data_test_block, batch_size=batch_size) #, num_workers=4, persistent_workers=True
    
    # return a random batch from the dataset
    def get_batch(self, data, batch_size, block_size):
        if data == "train": data = self.train_data
        else: data = self.test_data
        
        idx = torch.randint(len(data)-block_size-1, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in idx])
        y = torch.stack([data[i+1:i+block_size+1] for i in idx])
        return x, y
        

if __name__ == "__main__":
    data = Data()
    dataLoader = data.getDataLoader(8)
    features, labels = next(iter(dataLoader))
    print(f"Features:\n {features}, Labels\n: {labels}")
    tokenizer = data.tokenizer
    
    print(f"First 500 token of training data:\n\n{tokenizer.decode(data.train_data[:500])}")
    print(f"Train batch{data.get_batch('train', 4, 8)}")
        