import torch
import lightning as L

from Data import Data
from LTrainer import LiLanguageModel
from DecoderTransformer import DecoderTransformer


data = Data(bite_pair_encoding=100)
tokenizer = data.tokenizer

model = LiLanguageModel.load_from_checkpoint("lightning_logs\\Stable\\checkpoints\\epoch=9-step=89270.ckpt")
idx = torch.zeros((1,1), dtype = torch.int64, device=model.device)
gen_token = model.generate(idx, 100)
gen_text = tokenizer.decode(gen_token)
print(gen_text)