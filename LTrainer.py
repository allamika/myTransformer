import torch
import torchmetrics
import lightning as L
from lightning.pytorch import loggers as pl_loggers

import Loss
from DecoderTransformer import DecoderTransformer


class LiLanguageModel(L.LightningModule):
  def __init__(self, vocab_size, hyper_param):
    super().__init__()
    self.save_hyperparameters()
    self.block_size = hyper_param['block_size']
    self.lr = hyper_param['lr']
    self.LM = DecoderTransformer(vocab_size, hyper_param['block_size'], hyper_param['n_embd'], hyper_param['nb_head'])
    self.accuracy =torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size)

  def training_step(self, batch):
    x,y = batch
    out = self.LM(x)
    B, T, C = out.shape
    loss = Loss.eval_loss(out.view(B*T, C), y.view(B*T))

    self.log("loss", loss)
    return loss
  
  def test_step(self, batch):
    x,y = batch
    out = self.LM(x)
    
    B, T, C = out.shape
    outv = out.view(B*T, C)
    yv = y.view(B*T)
    test_loss = Loss.eval_loss(outv, yv)
    test_acc = self.accuracy(torch.argmax(outv, dim=1), yv)
    self.log("test_loss", test_loss)
    self.log("test_acc", test_acc)

  def validation_step(self, batch):
    x,y = batch
    out = self.LM(x)
    B, T, C = out.shape
    valid_loss = Loss.eval_loss(out.view(B*T, C), y.view(B*T))
    self.log("validation_loss", valid_loss)

  def on_fit_end(self):
    self.logger.log_metrics({'Test': 0})

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.LM.parameters(), lr=self.lr)
    return optimizer
  
  def generate(self,idx, max_new_tokens):
    return self.LM.generate(idx, max_new_tokens)
    
  

if __name__  == "__main__":
    import datetime
    from Data import Data
    
    time = datetime.datetime.now().__format__('%Y-%m-%dT%X').replace(":",".")
    hyper_param = {
        "batch_size": 64,
        "block_size" : 32,
        "n_embd" : 512,
        "nb_head" : 8,
        "lr" : 1e-3,
        "cut" : 1,
        "nbReduction" : 100,
        "nbepoch" : 10
        }
    

    data = Data(bite_pair_encoding=hyper_param["nbReduction"])
    tokenizer = data.tokenizer
    dataLoaderTrain, dataLoaderValid, dataLoaderTest = data.getDataLoaders(hyper_param['block_size'], hyper_param['batch_size'], cut=hyper_param['cut'])    
    
    lilm = LiLanguageModel(tokenizer.vocab_size(),  hyper_param)

    versionName = (f'DT_b{hyper_param['block_size']}e{hyper_param['n_embd']}h{hyper_param['nb_head']}_'
                   f'T_b{hyper_param['batch_size']}lr{hyper_param['lr']}_'
                   f'D_c{hyper_param['cut']}r{hyper_param['nbReduction']}_'
                   f'Ti_{time}')

    logger = pl_loggers.TensorBoardLogger(save_dir=".", version=versionName, name='lightning_logs')

    trainer = L.Trainer(logger=logger, max_epochs=hyper_param['nbepoch'], accelerator="auto", devices="auto", enable_checkpointing=True)
    trainer.test(lilm,dataloaders=dataLoaderTest)
    trainer.validate(lilm, dataloaders=dataLoaderValid)
    trainer.fit(model=lilm, train_dataloaders=dataLoaderTrain, val_dataloaders=dataLoaderValid)
    trainer.test(lilm,dataloaders=dataLoaderTest)
    
