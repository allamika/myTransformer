import torch
import lightning as L
from lightning.pytorch import loggers as pl_loggers

import Loss

class LiLanguageModel(L.LightningModule):
  def __init__(self, LM):
    super().__init__()
    self.LM = LM

  def training_step(self, batch):
    x,y = batch
    out = self.LM(x)
    loss = Loss.eval_loss(out, y)

    self.log("loss", loss)
    return loss
  
  def test_step(self, batch):
    x,y = batch
    out = self.LM(x)
    test_loss = Loss.eval_loss(out, y)
    self.log("test_loss", test_loss)

  def validation_step(self, batch):
    x,y = batch
    out = self.LM(x)
    test_loss = Loss.eval_loss(out, y)
    self.log("validation_loss", test_loss)



  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.LM.parameters(), lr=1e-3)
    return optimizer
  

if __name__  == "__main__":
    from Data import Data
    from DecoderTransformer import DecoderTransformer

    data = Data()
    tokenizer = data.tokenizer
    block_size = 32
    batch_size = 32
    dataLoaderTrain, dataLoaderValid, dataLoaderTest = data.getDataLoaders(block_size, batch_size, cut=0.1)

    lm = DecoderTransformer(tokenizer.vocab_size(), block_size)
    lilm = LiLanguageModel(lm)

    versionName = 'Test'
    logger = pl_loggers.TensorBoardLogger(save_dir=".", version=versionName, name='lightning_logs')

    trainer = L.Trainer(logger=logger, max_epochs=10, accelerator="auto", devices="auto", enable_checkpointing=True)
    trainer.test(lilm,dataloaders=dataLoaderTest)
    trainer.fit(model=lilm, train_dataloaders=dataLoaderTrain, val_dataloaders=dataLoaderValid)
    trainer.test(lilm,dataloaders=dataLoaderTest)
    
