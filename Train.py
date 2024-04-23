import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt

from pprint import pprint
from tqdm import tqdm


from Data import Data
from Loss import estimate_loss, eval_loss


def train(m, data, lr=1e-3, total_batch = 5000, block_size = 8, batch_size = 32, display_step = 500):
    print("---Training---")
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr)
    list_esti_loss = []
    
    for step in (pbar:=tqdm(range(total_batch), postfix=estimate_loss(m, data))):
        m.train()
        
        xb, yb = data.get_batch("train", batch_size, block_size)

        out = m(xb)
        
        optimizer.zero_grad(set_to_none=True)
        loss = eval_loss(out, yb)
        
        loss.backward()
        optimizer.step()
        
        if step % display_step == 0:
            estimated_loss = estimate_loss(m, data)
            pbar.set_postfix({"Loss": estimated_loss})
            list_esti_loss.append(estimated_loss)
    
    pprint(list_esti_loss)
    plt.plot(list(map(lambda x: x.get("train"), list_esti_loss)), label="Train")
    plt.plot(list(map(lambda x: x.get("test"), list_esti_loss)), label="Test")
    plt.legend()
    plt.show()
