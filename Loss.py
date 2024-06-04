from torch.nn import functional as F

#return a dict containing estimation of the model cross_entropy loss estimation on train and test on 100 iteration
#return format: dict{'train': train_loss(int), 'test': test_loss(int)}
def estimate_loss(model, data):
  iter_estimation = 100
  model.eval()
  loss_train ,loss_test = 0,0
  for i in range(iter_estimation):
    loss_train += eval_batch_loss(model, data.get_batch("train", 8, 8)).item()
    loss_test += eval_batch_loss(model, data.get_batch("test", 8, 8)).item()
  return {'train': loss_train/iter_estimation, 'test': loss_test/iter_estimation}

#return the cross_entropy loss of the model on a batch
def eval_batch_loss(model, batch):
  xb, yb = batch
  out = model(xb)
  return eval_loss(out, yb)

#return the cross_entropy loss of the model btw the two inputs, out and yb
def eval_loss(out, yb):
  loss = F.cross_entropy(out, yb)
  return loss