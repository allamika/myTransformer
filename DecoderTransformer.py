import torch
import torch.nn as nn
from torch.nn import functional as F

dropout = 0.2
n_embd = 64
nb_head = 4



class DecoderTransformer(nn.Module):

  def __init__(self, vocab_size, block_size):
    super().__init__()
    self.block_size = block_size
    self.token_embeding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.ma_head = MultiHeadAttention(n_embd, nb_head, n_embd//nb_head)
    self.ff = FeedForward(n_embd)
    self.blocks = nn.Sequential(
        Block(n_embd, n_head=nb_head),
        Block(n_embd, n_head=nb_head),
        Block(n_embd, n_head=nb_head),
        Block(n_embd, n_head=nb_head)
    )
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx):
    B,T = idx.shape

    tok_emb = self.token_embeding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T, device=idx.get_device()))
    x = tok_emb + pos_emb
    
    x = self.blocks(x)
    logits = self.lm_head(x)

    return logits

  def generate(self, idx, max_new_tokens):
    for _i in range(max_new_tokens):
      idx_cond = idx[:, -self.block_size:]
      #logit prediction (proba for each token to be the following)
      next_logits = self(idx_cond)
      next_logits = next_logits[:,-1,:]
      probs = F.softmax(next_logits, dim=-1)
      #select the token using proba
      next_token = torch.multinomial(probs, 1)
      #add the token to the generation
      idx = torch.cat((idx, next_token), dim=1)
    return idx


class Block(nn.Module):
  def __init__(self, n_emb, n_head):
    super().__init__()
    head_size = n_emb//n_head
    self.MH = MultiHeadAttention(n_emb, n_head, head_size)
    self.FF = FeedForward(n_emb)
    self.LN1 = nn.LayerNorm(n_emb)
    self.LN2 = nn.LayerNorm(n_emb)

  def forward(self, x):
    x = x + self.MH(self.LN1(x)) # residual connection
    out = x + self.FF(self.LN2(x)) # residual connection
    return out


class FeedForward(nn.Module):
  def __init__(self, n_emb):
    super().__init__()
    self.linear = nn.Linear(n_emb, n_emb*4)
    self.proj = nn.Linear(n_emb*4, n_emb) # projection layer for residual connection
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = F.relu(self.linear(x))
    out = self.proj(out)
    out = self.dropout(out)
    return out


class MultiHeadAttention(nn.Module):
  def __init__(self, n_emb, n_head, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(n_emb, head_size) for _i in range(n_head)])
    self.proj = nn.Linear(n_emb, n_emb) # projection layer for residual connection
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([head(x) for head in self.heads], dim=-1)
    out = self.proj(out)
    out = self.dropout(out)
    return out


class Head(nn.Module):
  def __init__(self, emb, head_size):
    super().__init__()
    self.key = nn.Linear(emb, head_size, bias=False)
    self.query = nn.Linear(emb, head_size, bias=False)
    self.value = nn.Linear(emb, head_size, bias=False)
    self.dropout = nn.Dropout(dropout)


  def forward(self, x):
    _B, T, C = x.shape

    k = self.key(x) # what i propose  (B,T,H)
    q = self.query(x) # what i ask for  (B,T,H)
    v = self.value(x) # what i communicate (information)  (B,T,H)

    wei = q @ k.transpose(-2, -1) # compatibiliy between tokens  (B,T,T)
    wei = wei / C**0.5  # prevent softmax to converge to one hot
    tril = torch.tril(torch.ones(T,T,device=x.get_device()))
    wei = wei.masked_fill(tril==0, float('-inf')) # force to know only about past tokens (decoder block)
    wei = F.softmax(wei, dim=-1) # interaction strength between tokens

    wei = self.dropout(wei)

    out = wei @ v

    return out



if __name__ == "__main__":
    from Data import Data
    
    data = Data()
    tokenizer = data.tokenizer
    
    m = DecoderTransformer(tokenizer.vocab_size(), 32)

    idx = torch.zeros((1,1), dtype = torch.int64)
    gen_token = m.generate(idx, 10)
    gen_text = tokenizer.decode(gen_token[0])
    print(gen_text)
