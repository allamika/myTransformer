import torch
import lightning as L
from lightning.pytorch import loggers as pl_loggers

import Loss


class LiLanguageModel(L.LightningModule):
  def __init__(self, LM):
    super().__init__()
    self.save_hyperparameters()
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
  
  def generate(self,idx, max_new_tokens):
    return self.LM.generate(idx, max_new_tokens)
    
  

if __name__  == "__main__":
    import datetime
    from Data import Data
    from DecoderTransformer import DecoderTransformer
    
    time = datetime.datetime.now().__format__('%Y-%m-%dT%X').replace(":",".")
    batch_size = 32
    block_size = 32
    n_embd = 384
    nb_head = 6
    lr = 1e-3
    cut = 1
    nbReduction = 0

    data = Data(bite_pair_encoding=nbReduction)
    tokenizer = data.tokenizer
    dataLoaderTrain, dataLoaderValid, dataLoaderTest = data.getDataLoaders(block_size, batch_size, cut=cut)    
    
    lm = DecoderTransformer(tokenizer.vocab_size(), block_size, n_embd, nb_head)
    lilm = LiLanguageModel(lm)

    versionName = f'DT_b{block_size}e{n_embd}h{nb_head}_T_b{batch_size}lr{lr}_D_c{cut}r{nbReduction}_Ti_{time}'
    logger = pl_loggers.TensorBoardLogger(save_dir=".", version=versionName, name='lightning_logs')

    trainer = L.Trainer(logger=logger, max_epochs=10, accelerator="auto", devices="auto", enable_checkpointing=True)
    trainer.test(lilm,dataloaders=dataLoaderTest)
    trainer.fit(model=lilm, train_dataloaders=dataLoaderTrain, val_dataloaders=dataLoaderValid)
    trainer.test(lilm,dataloaders=dataLoaderTest)
    
