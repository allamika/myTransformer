import torch
import torch.nn as nn
from torch.nn import functional as F

import Loss

class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.token_embeding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx):
    logits = self.token_embeding_table(idx)

    return logits

  def generate(self, idx, max_new_tokens):
    for _i in range(max_new_tokens):
      #logit prediction (proba for each token to be the following)
      next_logits = self(idx[-1])
      probs = F.softmax(next_logits, dim=-1)
      #select the token using proba
      next_token = torch.multinomial(probs, 1)
      #add the token to the generation
      idx = torch.concatenate((idx, next_token))
    return idx

if __name__ == "__main__":
    from Data import Data
    data = Data()
    tokenizer = data.tokenizer
    
    m = BigramLanguageModel(tokenizer.vocab_size())
    print(Loss.estimate_loss(m, data))
    
    idx = torch.zeros((1,1), dtype = torch.int64)
    gen_token = m.generate(idx, 100)
    gen_text = tokenizer.decode(gen_token)
    print(gen_text)