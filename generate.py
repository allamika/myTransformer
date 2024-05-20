import torch
import lightning as L

from Data import Data
from LTrainer import LiLanguageModel
from DecoderTransformer import DecoderTransformer


data = Data()
tokenizer = data.tokenizer

model = LiLanguageModel.load_from_checkpoint("lightning_logs\\Stable\\checkpoints\\epoch=9-step=28230.ckpt")
idx = torch.zeros((1,1), dtype = torch.int64, device=model.device)
gen_token = model.generate(idx, 1000)
gen_text = tokenizer.decode(gen_token)
print(gen_text)