import torch


from Data import Data
from Train import train
from DecoderTransformer import DecoderTransformer


bite_pair_encoding = 100
data  = Data(bite_pair_encoding=bite_pair_encoding)
tokenizer = data.tokenizer


lr = 1e-4
total_batch = 50000
batch_size = 64
block_size = 16

m = DecoderTransformer(tokenizer.vocab_size()+bite_pair_encoding, block_size)

train(m, data, lr = lr, total_batch=total_batch,  batch_size=batch_size, block_size=block_size)

idx = torch.zeros((1,1), dtype = torch.int64)
gen_token = m.generate(idx, 500)
gen_text = tokenizer.decode(gen_token[0])
print(gen_text)