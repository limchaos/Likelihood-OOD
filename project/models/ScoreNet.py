import torch, einops
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import repeat


class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x * self.W * 2 * np.pi
    return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
 
  

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    def __init__(self, dim=[2048, 2048, 2048, 2048], feat_dim=2, embed_dim=256):
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(dim=embed_dim),
             nn.Linear(embed_dim, embed_dim))
        
        self.dense1 = Dense(embed_dim, dim[0])
        self.linear1 = Dense(feat_dim, dim[0])
        self.bn1 = nn.BatchNorm1d(dim[0])
        
        self.dense2 = Dense(embed_dim, dim[1])
        self.linear2 = Dense(dim[0], dim[1])
        self.bn2 = nn.BatchNorm1d(dim[1])
        
        self.dense3 = Dense(embed_dim, dim[2])
        self.linear3 = Dense(dim[1], dim[2])
        self.bn3 = nn.BatchNorm1d(dim[2])
        
        self.dense4 = Dense(embed_dim, dim[3])
        self.linear4 = Dense(dim[2], dim[3])
        self.bn4 = nn.BatchNorm1d(dim[3])
        
        self.dense_out = Dense(embed_dim, feat_dim)
        self.linear_out = Dense(dim[0], feat_dim)

        # The swish activation function
        self.act = torch.nn.SiLU()

    def forward(self, x, t, p):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        h = self.linear1(x) + self.dense1(embed)
        #h = self.bn1(h)
        h = self.act(h)
        h = self.linear2(h) + self.dense2(embed)
        #h = self.bn2(h)
        h = self.act(h)
        h = self.linear3(h) + self.dense3(embed)
        #h = self.bn3(h)
        h = self.act(h)
        h = self.linear4(h) + self.dense4(embed)
        #h = self.bn4(h)
        h = self.act(h)
        h = self.linear_out(h) + self.dense_out(embed)

        #h = h / self.marginal_prob_std(t)[:, None]
        return h
  
class DenseAddEmbed(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim, embed_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
    self.embed = nn.Linear(embed_dim, output_dim)
  def forward(self, x, emb):
    x = self.dense(x) + self.embed(emb)
    return x

  
class ScoreNet_V2(nn.Module):
    def __init__(self, 
                 layer_dim=2048, 
                 feat_dim=768, 
                 activation = torch.nn.SiLU(), 
                 embed_dim=256, 
                 layers = 3, 
                 normalization = nn.LayerNorm, 
                 blocks = DenseAddEmbed, 
                 layout = 1,
                 ):
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(dim=embed_dim),
             nn.Linear(embed_dim, embed_dim))
        self.to_encoder = blocks(feat_dim, layer_dim, embed_dim)
        self.act = activation
        self.norm = normalization(layer_dim)
        if layers > 2:
            if layers % 2 == 0:
                first = layers//2
                second = layers//2
            else:
                first = layers//2 
                second = layers//2 + 1
            dims = np.concatenate([np.linspace(layer_dim, layer_dim*layout, first, dtype=int), 
                                   np.linspace(layer_dim*layout, feat_dim, second, dtype=int)])
        else:
            dims = [layer_dim]*(layers-1) + [feat_dim]
        self.encoder = nn.ModuleList(
           [
              nn.Sequential(
                  blocks(in_size, out_size, embed_dim),
                  activation,
                  normalization(out_size),
                )
                for in_size, out_size in zip(dims[:-1], dims[1:])
           ]
        )
        
    def forward(self, x, t, p):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        x = self.norm(self.act(self.to_encoder(x, embed)))
       
        for layer in self.encoder:
            feature, act, norm = layer
            x = norm(act(feature(x, embed)))
        return x


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module



class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param mid_channels: the number of middle channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    """

    def __init__(
        self,
        channels,
        mid_channels,
        emb_channels,
        dropout,
        activation,
        use_context=False,
        context_channels=512
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout

        self.in_layers = nn.Sequential(
            nn.LayerNorm(channels),
            activation,
            nn.Linear(channels, mid_channels, bias=True),
        )

        self.emb_layers = nn.Sequential(
            activation,
            nn.Linear(emb_channels, mid_channels, bias=True),
        )

        self.out_layers = nn.Sequential(
            nn.LayerNorm(mid_channels),
            activation,
            nn.Dropout(p=dropout),
            zero_module(
                nn.Linear(mid_channels, channels, bias=True)
            ),
        )

        self.use_context = use_context

        self.context_layers = nn.Sequential(
            activation,
            nn.Linear(context_channels, mid_channels, bias=True),
        )

    def forward(self, x, emb, context):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        
        # context
        context_out = self.context_layers(context)
        #print(torch.sum(context_out))
        h_con = h + emb_out + context_out
    
        # no context
        h = h + emb_out

        if self.use_context:
            #print(self.use_context)
            h = self.out_layers(h_con)
        else:
            #print(self.use_context)
            h = self.out_layers(h)
        return x + h


class SimpleMLP(nn.Module):
    """
    The full skip network with timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        time_embed_dim,
        model_channels,
        bottleneck_channels,
        out_channels,
        num_res_blocks,
        activation=nn.SiLU(),
        dropout=0,
        use_context=False,
        context_channels=512
    ):
        super().__init__()

        self.image_size = 1
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.context_channels = context_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            activation,
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.context_gf = GaussianFourierProjection(dim=context_channels) #nn.Embedding(1001, context_channels)

        self.context_embed = nn.Sequential(
            nn.Linear(context_channels, context_channels),
            activation,
            nn.Linear(context_channels, context_channels),
        )

        self.input_proj = nn.Linear(in_channels, model_channels)
        self.use_context = use_context
        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
                bottleneck_channels,
                time_embed_dim,
                dropout,
                use_context=self.use_context,
                activation = activation,
                context_channels=context_channels
            ))

        self.res_blocks = nn.ModuleList(res_blocks)

        self.out = nn.Sequential(
            nn.LayerNorm(model_channels, eps=1e-6),
            activation,
            zero_module(nn.Linear(model_channels, out_channels, bias=True)),
        )

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        x = x.squeeze()
        x = self.input_proj(x)
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        context = self.context_gf(context)
        context = self.context_embed(context)

        for block in self.res_blocks:
            x = block(x, emb, context)
        x = self.out(x)#.unsqueeze(-1).unsqueeze(-1)
        return x

