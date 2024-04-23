from torch.nn import functional as F

def estimate_loss(model, data):
  iter_estimation = 100
  model.eval()
  loss_train ,loss_test = 0,0
  for i in range(iter_estimation):
    loss_train += eval_batch_loss(model, data.get_batch("train", 8, 8)).item()
    loss_test += eval_batch_loss(model, data.get_batch("test", 8, 8)).item()
  return {'train': loss_train/iter_estimation, 'test': loss_test/iter_estimation}

def eval_batch_loss(model, batch):
  xb, yb = batch
  out = model(xb)
  return eval_loss(out, yb)

def eval_loss(out, yb):
  out = out.transpose(1,2)
  loss = F.cross_entropy(out, yb)
  return loss