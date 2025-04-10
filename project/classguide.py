import torch
import numpy as np
import tqdm
import torch.nn as nn

  

class guided_score:
    """This class takes a score model s(x,t) and a time dependent classifier model p(y|x,t), and provides a score that is guided by the classifier.
    It overrides the forward call of the score model to return the ordinary score plus the gradient of log-probability of classifier mutiplied by the constant gamma."""

    def __init__(self,score_model,classifier,gamma=10):
        self.classifier=classifier
        self.score_model=score_model
        self.logsoftmax=nn.LogSoftmax(dim=1)
        self.t0_samples=None
        self.gamma=gamma

    def eval(self):
       self.score_model.eval()
       self.classifier.eval()

    def __call__(self,samples,t,p):
        x=samples
        y=torch.argmax(self.classifier(self.t0_samples,torch.zeros(x.shape[0],device=x.device)),dim=1).detach()

        with torch.enable_grad():
            x.requires_grad=True
            pred=self.logsoftmax(self.classifier(x,t))[torch.arange(0,y.shape[0],dtype=torch.long),y]
            g=torch.autograd.grad((pred.sum(),),(x,))
       
        s= self.score_model(x,t,p) + self.gamma*g[0]

        return s

class set_clean_samples:
    """This context manager is used when calling a guided_score model since we need to keep track of the noise-free images (at t=0)"""
    def __init__(self, model,batch):
        self.model=model
        model.t0_samples=batch.clone()
    def __enter__(self):
        return None
    def __exit__(self, type, value, traceback):
        self.model.t0_samples=None   
       
ce_loss=torch.nn.CrossEntropyLoss()

class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
     
def loss_fn(model, x, labels, sde, eps=1e-5):
  """The loss function for training a time dependent classifier.
  Based on the loss function of score models, but with cross entropy on labels of (in-distribtuion) data as objective.
  """

  # Perturbing data according to the diffusion process

  t = torch.rand(x.shape[0], device=x.device) * (sde.T - eps) + eps
  z = torch.randn_like(x)
  mean, std = sde.marginal_prob(x, t)
  perturbed_x = mean + std[:, None] * z
  y = model(perturbed_x, t)
  loss = ce_loss(y,labels)

  return loss

def train_classifier(dataset_labeled, model, loss_fn, optimizer, n_epochs, batch_size, device, num_workers=0, verbose=True, tw=None, lrs=True):
    """Training a time dependent classifier"""

    if verbose:
        print(f'Training for {n_epochs} epochs...')
      
    data_loader = torch.utils.data.DataLoader(dataset_labeled, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if verbose:
        epochs = tqdm.trange(n_epochs)
    else:
        epochs = range(n_epochs)
    if lrs:
        lr = optimizer.param_groups[0]['lr']
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(data_loader), epochs=n_epochs)
    scaler = torch.cuda.amp.GradScaler()
    avg_epoch_loss = []
    model.train()
    for epoch in epochs:
        avg_loss = 0.
        num_items = 0
        epoch_loss = []
        for x,y in data_loader:
            x,y = x.to(device),y.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                loss = loss_fn(model, x, y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if lrs:
                lr_scheduler.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            epoch_loss.append(loss.item())
        epoch_loss = np.mean(epoch_loss)
        avg_epoch_loss.append(epoch_loss)
        if tw is not None:
            tw.add_scalar('Loss/train', epoch_loss, epoch)
        # Print the averaged training loss so far.
        if verbose:
            epochs.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    return avg_epoch_loss


class ClassifierNet(nn.Module):
    """Neural network for time dependent classifier. Based on 'Scorenet'. """

    def __init__(self, dim=[200, 200], feat_dim=768, embed_dim=256,num_class=1000):
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
             nn.Linear(embed_dim, embed_dim))
        
        self.dense1 = Dense(embed_dim, dim[0])
        self.linear1 = Dense(feat_dim, dim[0])
        self.bn1 = nn.BatchNorm1d(dim[0])
        
        self.dense2 = Dense(embed_dim, dim[1])
        self.linear2 = Dense(dim[0], dim[1])
        self.bn2 = nn.BatchNorm1d(dim[1])

        self.dense3 = Dense(embed_dim, dim[1])
        self.linear3 = Dense(dim[1], dim[1])
        self.bn3 = nn.BatchNorm1d(dim[1])
        
        self.dense4 = Dense(embed_dim, dim[1])
        self.linear4 = Dense(dim[1], dim[1])
        self.bn4 = nn.BatchNorm1d(dim[1])
        #self.dense_out = Dense(embed_dim, 1)
        self.linear_out = Dense(dim[1], num_class)

        # The swish activation function
        self.act = torch.nn.ReLU()

        self.num_class=num_class

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        h = self.linear1(x) + self.dense1(embed)
        h = self.bn1(h)
        h = self.act(h)
        h = self.linear2(h) + self.dense2(embed)
        h = self.bn2(h)
        h = self.act(h)
        h = self.linear3(h) + self.dense3(embed)
        h = self.bn3(h)
        h = self.act(h)
        h = self.linear4(h) + self.dense4(embed)
        h = self.bn4(h)
        h = self.act(h)
        #h = self.linear_out(h) + self.dense_out(embed)
        h= self.linear_out(h)

        return h