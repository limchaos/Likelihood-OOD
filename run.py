import torch
import project.utils as utils
import project.train as train
import project.losses as losses

from project.sde_lib import VESDE, VPSDE, subVPSDE
from project.models.ema import ExponentialMovingAverage
from project.models.ScoreNet import *
from scipy.special import softmax
from sklearn.covariance import EmpiricalCovariance

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('encoder', type = str)
    parser.add_argument('lr', type=float)
    parser.add_argument('n_epochs', type=int)
    parser.add_argument('weight_decay', type=float, default=0)
    parser.add_argument('train_data_path', type = str)
    parser.add_argument('ckpt_dir', type = str)

    parser.add_argument('-logit', action='store_true')
    parser.add_argument('-norm', action='store_true')
    return parser.parse_args()

args = parse_args()
print(args)
feature_dir = 'features'
ckpt_dir = args.ckpt_dir
encoder= args.encoder #'swin'
logit = args.logit
norm = args.norm
pca = True

device = 'cuda:0' 
num_workers = 6

#network
bottleneck_channels=1536
num_res_blocks= 12
time_embed_dim=256
context_channels = 256 
dropout = 0

# optim
lr= args.lr
beta1 = 0.99
eps = 1e-8
weight_decay = args.weight_decay
warmup = 2000
grad_clip = 1
ema_rate=0.9999

# train
n_epochs = args.n_epochs # 60
batch_size = 4096
continuous = True
reduce_mean = True
likelihood_weighting = False
beta_min = 0.2 #0.15
beta_max = 20 #19.5
sigma_min = 0.01
sigma_max = 50

if logit:
   con = 'con'
else:
   con = 'uncon'


# train_data_path = 'nontumor_train.pkl' #
# reference_data_path = 'nontumor_test.pkl'
# eval_data_paths = [
#     'tumor_test.pkl', 
#     ]

train_data_path = args.train_data_path #'imagenet_all.pkl' 
reference_data_path = 'imagenet2012_val_list.pkl'
eval_data_paths = [
    'openimage_o.pkl', 
    'texture.pkl',
    'iNaturalist.pkl', 
    'imagenet-o.pkl',
    ]

train_blob = utils.load_data_blob(feature_dir, encoder, train_data_path)
reference_blob = utils.load_data_blob(feature_dir, encoder, reference_data_path)
eval_blobs = [utils.load_data_blob(feature_dir, encoder, eval_data_path) for eval_data_path in eval_data_paths]


if logit:
    scale = 1
    weight_path = 'fc.pkl'
    w,b = utils.load_data_blob(feature_dir, encoder, weight_path)
    
    train_blob['label'] =  1 + np.argmax(softmax(train_blob['data'] @ w.T + b, axis=1) * scale, axis=1).reshape(-1, 1).astype(np.int64)
    reference_blob['label'] = 1 + np.argmax(softmax(reference_blob['data'] @ w.T + b, axis=1) * scale, axis=1).reshape(-1, 1).astype(np.int64)
    
    for eval_blob in eval_blobs:
        eval_blob['label'] = 1 + np.argmax(softmax(eval_blob['data'] @ w.T + b, axis=1) * scale, axis=1).reshape(-1, 1).astype(np.int64)
else:
    
    train_blob['label'] =  - np.ones(train_blob['data'].shape[0]).reshape(-1, 1)
    reference_blob['label'] = - np.ones(reference_blob['data'].shape[0]).reshape(-1, 1)
    for eval_blob in eval_blobs:
        eval_blob['label'] = - np.ones(eval_blob['data'].shape[0]).reshape(-1, 1)
    
if pca:
    print(pca)
    DIM =  768
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(train_blob['data'])
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[:DIM]]).T)
    
    train_blob['data'] = np.float32(np.matmul(train_blob['data'], NS))
    print(train_blob['data'].shape[-1])
    reference_blob['data'] = np.float32(np.matmul(reference_blob['data'], NS))
    print(train_blob['data'].shape[-1])
    for eval_blob in eval_blobs:
        eval_blob['data'] = np.float32(np.matmul(eval_blob['data'], NS))
        print(eval_blob['data'].shape[-1])
        
    feat_dim=train_blob['data'].shape[-1]  
    
if norm:
    
    means = np.mean(train_blob['data'], axis=0)
    stds = np.std(train_blob['data'], axis=0)
    normalizer = lambda x: (x - means)/ stds
    
#     meanser = lambda x : np.mean(x, axis=1, keepdims=True)
#     stdser = lambda x : np.std(x, axis=1, keepdims=True)
#     normalizer = lambda x: (x - meanser(x))/ stdser(x)
    
    prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))

    train_blob['data'] = prepos_feat(train_blob['data'])
    reference_blob['data'] = prepos_feat(reference_blob['data']) 
    

    for eval_blob in eval_blobs:
        eval_blob['data'] = prepos_feat(eval_blob['data'])

feat_dim=train_blob['data'].shape[-1]

sde_str = 'subVPSDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde_str.lower() == 'vesde':
  sde = VESDE(sigma_min=sigma_min, sigma_max=sigma_max, N=1000)
  sampling_eps = 1e-5
elif sde_str.lower() == 'vpsde':
  sde = VPSDE(beta_min=beta_min, beta_max=beta_max, N=1000)
  sampling_eps = 1e-3
elif sde_str.lower() == 'subvpsde':
  sde = subVPSDE(beta_min=beta_min, beta_max=beta_max, N=1000)
  sampling_eps = 1e-3 

score_model = SimpleMLP(in_channels=feat_dim,
        time_embed_dim=time_embed_dim,
        model_channels=1536,
        bottleneck_channels=bottleneck_channels,
        out_channels=feat_dim,
        num_res_blocks=num_res_blocks,
        activation=nn.SiLU(), #SiLU best
        dropout=dropout,
        use_context=logit,
        context_channels=context_channels)

score_model = score_model.to(device)

ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=ema_rate)
# optimizer = torch.optim.Adam(score_model.parameters(), lr=lr, betas=(beta1, 0.999), eps=eps,
#                          weight_decay=weight_decay)

optimizer = torch.optim.AdamW(score_model.parameters(), lr=lr, weight_decay=weight_decay)

state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

optimize_fn = losses.optimization_manager(lr, warmup, grad_clip)
train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                  reduce_mean=reduce_mean, continuous=continuous,
                                 likelihood_weighting=likelihood_weighting)

loss_fn = losses.get_sde_loss_fn(sde, train=True, reduce_mean=reduce_mean, continuous=continuous, 
                       likelihood_weighting=likelihood_weighting, eps=1e-5)



train_loss = train.train_con(train_blob['data'], train_blob['label'], score_model, loss_fn, optimizer, n_epochs, batch_size, device, num_workers=0, verbose=True, tw=None, lrs=True)
with open( 'checkpoint/' + ckpt_dir + '/' + con + '/' + sde_str + '_' + train_blob['info']['model'] + '.txt', 'w') as file:
    # Write some text to the file
    file.write(str(train_loss)+'\n')
    file.write(str(args))
torch.save(score_model.state_dict(), 'checkpoint/' + ckpt_dir + '/' + con + '/' + sde_str + '_' + train_blob['info']['model'] + '.pth')