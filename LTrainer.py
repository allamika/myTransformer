import torch
import lightning as L

import Loss

class LiLanguageModel(L.LightningModule):
  def __init__(self, LM):
    super().__init__()
    self.LM = LM

  def training_step(self, batch):
    x,y = batch
    out = self.LM(x)
    loss = Loss.eval_loss(out, y)

    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.LM.parameters(), lr=1e-3)
    return optimizer
  

if __name__  == "__main__":
    from Data import Data
    from DecoderTransformer import DecoderTransformer

    data = Data()
    tokenizer = data.tokenizer
    block_size = 8
    dataLoader = data.getDataLoader(block_size)

    lm = DecoderTransformer(tokenizer.vocab_size(), block_size)

    lilm = LiLanguageModel(lm)
    trainer = L.Trainer(max_epochs=1, accelerator="auto", devices="auto")
    trainer.fit(model=lilm, train_dataloaders=dataLoader)
