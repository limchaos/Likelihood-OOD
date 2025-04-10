
import torch
import project.utils as utils
import project.train as train

from project.sde_lib import VESDE, VPSDE, subVPSDE
from project.likelihood import get_likelihood_fn
from project.models.ScoreNet import *
from scipy.special import softmax
import pickle
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('encoder', type = str, help='1')
    parser.add_argument('ckpt_dir', type = str)
    parser.add_argument('-logit', action='store_true')
    parser.add_argument('-norm', action='store_true')
    return parser.parse_args()

args = parse_args()
print(args)
device = 'cuda:0' 
num_workers = 6
ckpt_dir = args.ckpt_dir


#network
bottleneck_channels=1536
num_res_blocks= 12
time_embed_dim=256
context_channels = 256 
dropout = 0
batch_size = 4096

# SDE
continuous = True
reduce_mean = True
likelihood_weighting = False
beta_min = 0.2 #0.15
beta_max = 20 #19.5
sigma_min = 0.01
sigma_max = 50

#data
logit = args.logit
norm = args.norm
pca = True
feature_dir = 'features'
encoder= args.encoder 
if logit:
   con = 'con'
else:
   con = 'uncon'
train_data_path = 'imagenet2012_train_random_200k.pkl' 
reference_data_path = 'imagenet2012_val_list.pkl'
eval_data_paths = [
    'openimage_o.pkl', 
    'texture.pkl',
    'iNaturalist.pkl', 
    'imagenet-o.pkl',
    ]

# train_data_path = 'nontumor_train.pkl' #
# reference_data_path = 'nontumor_test.pkl'
# eval_data_paths = [
#     'tumor_test.pkl', 
#     ]

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

if norm:
    
    means = np.mean(train_blob['data'], axis=0)
    stds = np.std(train_blob['data'], axis=0)
    normalizer = lambda x: (x - means) / (stds + 1e-8)
    
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


ckpt = torch.load('checkpoint/' + ckpt_dir + '/' + con + '/' + sde_str + '_' + train_blob['info']['model'] + '.pth', map_location=device)
score_model.load_state_dict(ckpt)
score_model.to(device)

batches = None
score_model.to(device)
score_model.use_context

likelihood_fn = get_likelihood_fn(sde)

eval_data = train.eval(
    train_blob['data'][:10],
    train_blob['label'][:10],
    reference_blob['data'],
    reference_blob['label'],
    [e['data'] for e in eval_blobs],
    [e['label'] for e in eval_blobs],
    score_model, likelihood_fn, batch_size, device, num_workers, batches=batches)

auc = np.round(eval_data['auc'], 4) * 100
fpr = np.round(eval_data['fpr'], 4) * 100

results = ' & '
for i in range(4):
    results += str(auc[i])
    results += ' & '
    results += str(fpr[i])
    results += ' & '
results += str(np.round(np.mean(auc), 2))
results += ' & '
results += str(np.round(np.mean(fpr), 2))

print(results)

with open( 'checkpoint/' + ckpt_dir + '/' + con + '/' + sde_str + '_' + train_blob['info']['model'] + '.txt', 'r') as file:
    train_log = file.read()

with open('eval/' + ckpt_dir + '/' + con + '.txt', 'a') as file:
    # Write some text to the file
    file.write(train_log + '\n')
    file.write(results + '\n')

with open('rdm/'+ encoder +'.pkl', 'wb') as f:
    pickle.dump(eval_data, f) 