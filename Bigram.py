import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning as L

import Loss

class LBigramLanguageModel(L.LightningModule):
  def __init__(self, bigram):
    super().__init__()
    self.bigram = bigram

  def training_step(self, batch):
    x,y = batch
    out = self.bigram(x)
    loss = Loss.eval_loss(out, y)

    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    return optimizer

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
      idx = torch.cat((idx, next_token))
    return idx.squeeze(1)

if __name__ == "__main__":
    from Data import Data
    data = Data()
    tokenizer = data.tokenizer
    
    m = BigramLanguageModel(tokenizer.vocab_size())
    
    idx = torch.zeros((1,1), dtype = torch.int64)
    gen_token = m.generate(idx, 100)
    gen_text = tokenizer.decode(gen_token)
    print(gen_text)

    dataLoader = data.getDataLoader(8)

    lm = LBigramLanguageModel(m)
    trainer = L.Trainer(max_epochs=10, accelerator="auto", devices="auto")
    trainer.fit(model=lm, train_dataloaders=dataLoader)
