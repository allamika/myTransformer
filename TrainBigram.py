import torch


from Data import Data
from Train import train
from Bigram import BigramLanguageModel

bite_pair_encoding = 10
data  = Data(bite_pair_encoding=bite_pair_encoding)
tokenizer = data.tokenizer

m = BigramLanguageModel(tokenizer.vocab_size()+bite_pair_encoding)

train(m, data)

idx = torch.zeros((1,1), dtype = torch.int64)
gen_token = m.generate(idx, 500)
gen_text = tokenizer.decode(gen_token)
print(gen_text)
