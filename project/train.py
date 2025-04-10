import pathlib
import pickle
import torch
import numpy as np
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import project.utils as utils
import functools
#import yaml
import argparse
#import optuna
import copy
import random
import matplotlib.ticker as ticker
import itertools 

from project.models.ScoreNet import ScoreNet_V2, DenseAddEmbed, SimpleMLP
from project.utils import parallelize

#from torch.utils.tensorboard import SummaryWriter

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import project.classguide as classguide


def loss_fn(model, x, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=1))
  return loss

def train_con(dataset, con, model, loss_fn, optimizer, n_epochs, batch_size, device, num_workers=0, verbose=True, tw=None, lrs=True):
    if verbose:
        print(f'Training for {n_epochs} epochs...')

    dataset = torch.utils.data.TensorDataset(torch.Tensor(dataset), torch.Tensor(con))   
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if verbose:
        epochs = tqdm.trange(n_epochs)
    else:
        epochs = range(n_epochs)
    if lrs:
        
        lr = optimizer.param_groups[0]['lr']
        #lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(data_loader), epochs=n_epochs)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, pct_start=0.05, cycle_momentum=False, anneal_strategy='cos', steps_per_epoch=len(dataset) // batch_size + 1, epochs=n_epochs)
    scaler = torch.cuda.amp.GradScaler()
    avg_epoch_loss = []
    model.train()
    for epoch in epochs:
        avg_loss = 0.
        num_items = 0
        epoch_loss = []
        for x, p  in data_loader:
            # 0 is unconditional, 1 is fully conditional
            #p = torch.tensor(np.random.binomial(1, 0.8, p.shape)) * p
            #p = p.type(torch.LongTensor)
            x = x.to(device)
            p = p.to(device)
            #with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            #with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            loss = loss_fn(model, x, p)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
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



def inference(dataset, con, model, ode_likelihood, batch_size, device, num_workers=0, batches=None, verbose=True, tw=None):
    
    dataset = torch.utils.data.TensorDataset(torch.Tensor(dataset), torch.Tensor(con))  
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    all_bpds = 0.
    all_items = 0
    score_id = []
    if verbose:
        data_iter = tqdm.tqdm(data_loader)
    else:
        data_iter = data_loader
    model.eval()
    batchcount = 0  
    for x, p in data_iter:
        x = x.to(device)
        p = p.to(device)
        if type(model) is classguide.guided_score: # when the score model is guided by a classifier
            with classguide.set_clean_samples(model, x): 
                bpd = ode_likelihood(model=model, x=x, p=p)
                #bpd = ode_likelihood(x, torch.Tensor([1]).cuda(), p)
        else:
            bpd = ode_likelihood(model=model, x=x, p=p)
            #bpd = ode_likelihood(x, torch.Tensor([1]).cuda(), p)
        all_bpds += bpd.sum()
        all_items += bpd.shape[0]
        score_id.append(bpd.detach().cpu().numpy())

        if verbose:
            data_iter.set_description("Average bits/dim: {:5f}".format(all_bpds / all_items))
        batchcount += 1

        if batches is not None and batchcount >= batches:
            break
    return np.concatenate(score_id)


def eval(dataset, dataset_con, reference_dataset, reference_dataset_con, ood_datasets, ood_datasets_con, model, ode_likelihood, batch_size, device, num_workers=0, recall=0.95, batches=None, verbose=True, tw=None):
    if verbose:
        print('Running eval. In-distribution data')
    score_id = inference(dataset, dataset_con, model, ode_likelihood, batch_size, device, num_workers, batches, verbose=verbose, tw=tw)
    if verbose:
        print('Running eval. Reference data')
    score_ref = inference(reference_dataset, reference_dataset_con, model, ode_likelihood, batch_size, device, num_workers, batches, verbose=verbose, tw=tw)
    ref_auc, _, _ = utils.auc(-score_ref, -score_id)
    ref_fpr, _ = utils.fpr_recall(-score_ref, -score_id, recall)
    if verbose:
        print(f'AUC: {ref_auc:.4f}, FPR: {ref_fpr:.4f}')
    score_oods = []
    auc_oods = []
    fpr_oods = []
    for i, (ood_dataset, ood_dataset_con) in enumerate(zip(ood_datasets, ood_datasets_con)):
        if verbose:
            print(f'Running eval. Out-of-distribution data {i+1}/{len(ood_datasets)}')
        score_ood = inference(ood_dataset, ood_dataset_con, model, ode_likelihood, batch_size, device, num_workers, batches, verbose=verbose, tw=tw)
        print("scores shape",score_ood.shape)
        score_oods.append(score_ood)
        auc_ood, _, _ = utils.auc(-score_ref, -score_ood)
        auc_oods.append(auc_ood)
        fpr_ood, _ = utils.fpr_recall(-score_ref, -score_ood, recall)
        fpr_oods.append(fpr_ood)
        if verbose:
            print(f'AUC: {auc_ood:.4f}, FPR: {fpr_ood:.4f}')

    return {'score': score_id, 
            'score_ref': score_ref,
            'ref_auc': ref_auc,
            'ref_fpr': ref_fpr,
            'score_oods': score_oods, 
            'auc': auc_oods, 
            'fpr': fpr_oods}

def train_new(dataset, train_step_fn, state, n_epochs, batch_size, device, num_workers=0, verbose=True, tw=None):
    if verbose:
        print(f'Training for {n_epochs} epochs...')
      
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if verbose:
        epochs = tqdm.trange(n_epochs)
    else:
        epochs = range(n_epochs)

    avg_epoch_loss = []
    for epoch in epochs:
        avg_loss = 0.
        num_items = 0
        epoch_loss = []
        for x in data_loader:
            x = x.to(device)
            loss = train_step_fn(state, x)
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

def train_new_con(dataset, con, train_step_fn, state, n_epochs, batch_size, device, num_workers=0, verbose=True, tw=None):
    if verbose:
        print(f'Training for {n_epochs} epochs...')

    dataset = torch.utils.data.TensorDataset(torch.Tensor(dataset), torch.Tensor(con))  
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if verbose:
        epochs = tqdm.trange(n_epochs)
    else:
        epochs = range(n_epochs)

    avg_epoch_loss = []
    for epoch in epochs:
        avg_loss = 0.
        num_items = 0
        epoch_loss = []
        # if epoch % 10 == 0:
        #     state['model'].use_context = False
        # else:
        #     state['model'].use_context = True
        for x, p in data_loader:
            x = x.to(device)
            p = p.to(device)
            loss = train_step_fn(state, x, p)
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


def plot(eval_data, ood_names, model_name, data_name, out_dir='figs', config=None, verbose=True):
    if verbose:
        print('Generating plots...')
    # Unpack eval_data
    score, score_ref = eval_data['score'], eval_data['score_ref']
    ref_auc, ref_fpr = eval_data['ref_auc'], eval_data['ref_fpr']
    train_loss = eval_data['train_loss']
    score_oods, auc_oods, fpr_oods = eval_data['score_oods'], eval_data['auc'], eval_data['fpr']

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # Adjust the size as needed
    fig.suptitle(f'{model_name} Evaluation')

    def add_shadow(ax, data): 
        l = ax.lines[-1]
        x = l.get_xydata()[:,0]
        y = l.get_xydata()[:,1]
        ax.fill_between(x,y, alpha=0.1)
        # Calculate and plot the mean
        mean_value = np.mean(data)
        line_color = l.get_color()
        ax.axvline(mean_value, color=line_color, linestyle=':', linewidth=1.5)
    # Subplot 1: KDE plots
    sns.kdeplot(data=score, bw_adjust=.2, ax=axs[0, 0], label=f'Training: {np.mean(score):.2f}')
    add_shadow(axs[0, 0], score)

    sns.kdeplot(data=score_ref, bw_adjust=.2, ax=axs[0, 0], label=f'Validation: {np.mean(score_ref):.2f}')
    add_shadow(axs[0, 0], score_ref)

    for ood_name, score_ood in zip(ood_names, score_oods):
        sns.kdeplot(data=score_ood, bw_adjust=.2, ax=axs[0, 0], label=f'{ood_name}: {np.mean(score_ood):.2f}')
        add_shadow(axs[0, 0], score_ood)
    axs[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(base=0.5))
    axs[0, 0].xaxis.set_minor_locator(ticker.MultipleLocator(base=0.1))
    axs[0, 0].set_title('Density Plots')
    axs[0, 0].set_xlabel('bits/dim')
    axs[0, 0].set_ylabel('Density')
    # axs[0, 0].set_xlim(6.5, 8)
    # axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Subplot 2: Bar chart for AUC and FPR
    x = np.arange(len(ood_names)+1)  # the label locations
    width = 0.35  # the width of the bars
    disp_auc = [ref_auc] + auc_oods
    disp_fpr = [ref_fpr] + fpr_oods
    rects1 = axs[0, 1].bar(x - width/2, disp_auc, width, label='AUC', alpha=0.6)
    rects2 = axs[0, 1].bar(x + width/2, disp_fpr, width, label='FPR', alpha=0.6)
    axs[0, 1].set_ylabel('Metric Value')
    axs[0, 1].yaxis.set_major_locator(ticker.MultipleLocator(base=0.1))
    axs[0, 1].yaxis.set_minor_locator(ticker.MultipleLocator(base=0.05))
    axs[0, 1].set_title(f'AUC and FPR Metrics\nMean AUC: {np.mean(disp_auc[1:]):.2f}, Mean FPR: {np.mean(disp_fpr[1:]):.2f}')
    axs[0, 1].set_xticks(x)
    names = [f'{name}\nACU: {auc:.2f}\nFPR: {fpr:.2f}' for name, auc, fpr in zip(['Training']+ood_names, disp_auc, disp_fpr)]
    axs[0, 1].set_xticklabels(names)
    axs[0, 1].grid(True)
    axs[0, 1].set_ylim([0, 1])
    axs[0, 1].legend()
    # add line at 0.5
    axs[0, 1].axhline(0.5, color='red', linestyle='--', linewidth=1.5) 

    # Subplot 3: Training loss over time
    axs[1, 0].plot(train_loss, label='Training Loss')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].set_title('Training Loss Over Time')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # Subplot 4: Configuration display
    if config is not None:
        config_text = "\n".join([f"{key}: {value}" for key, value in config.items()])
        axs[1, 1].text(0.5, 0.5, config_text, ha='center', va='center', fontsize=12, transform=axs[1, 1].transAxes)
        axs[1, 1].set_title('Configuration')
    axs[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout

    # Save the figure
    out_dir = pathlib.Path(out_dir) / model_name / data_name
    out_dir.mkdir(exist_ok=True, parents=True)
    filename = f"{model_name}_{data_name}_{int(np.mean(disp_auc[1:])*100)}.svg"
    plt.savefig(out_dir / filename, bbox_inches='tight')
    if verbose:
        plt.show()

def train_eval(config):
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    
    id = config['id']
    verbose = config['verbose']

    feature_dir = config['feature_dir']
    encoder=config['encoder']
    
    train_data_path = config['train_data_path']
    reference_data_path = config['reference_data_path']
    eval_data_paths = config['eval_data_paths']
    
    time_embed_dim = config['time_embed_dim']
    bottleneck_channels = config['bottleneck_channels']
    num_res_blocks = config['num_res_blocks']
    dropout = config['dropout']

    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    lr=config['lr']
    sigma = config['sigma']
    num_workers = config['num_workers']
    
    batches = config['batches']

    checkpoints = config['checkpoints']
    
    device = config['device']
    lrs = config['lrs']

    pathlib.Path(checkpoints).mkdir(exist_ok=True, parents=True)
    
    # tensorboard_dir = config['tensorboard_dir']+f'/{encoder}_id_{id}'
    # pathlib.Path(tensorboard_dir).mkdir(exist_ok=True, parents=True)
    # tw = SummaryWriter(tensorboard_dir)

    train_blob = utils.load_data_blob(feature_dir, encoder, train_data_path)
    reference_blob = utils.load_data_blob(feature_dir, encoder, reference_data_path)
    eval_blobs = [utils.load_data_blob(feature_dir, encoder, eval_data_path) for eval_data_path in eval_data_paths]


    marginal_prob_std_fn = functools.partial(utils.marginal_prob_std, sigma=sigma, device=device)
    diffusion_coeff_fn = functools.partial(utils.diffusion_coeff, sigma=sigma, device=device)
    feat_dim=train_blob['data'].shape[-1]


    score_model = SimpleMLP(marginal_prob_std=marginal_prob_std_fn,
        in_channels=feat_dim,
        time_embed_dim=time_embed_dim,
        model_channels=feat_dim,
        bottleneck_channels=bottleneck_channels,
        out_channels=feat_dim,
        num_res_blocks=num_res_blocks, 
        dropout=dropout)
    score_model = score_model.to(device)
    optimizer = torch.optim.Adam(score_model.parameters(), lr=lr)
    loss = functools.partial(loss_fn, marginal_prob_std=marginal_prob_std_fn)
    ode_likelihood = functools.partial(utils.ode_likelihood,
                                marginal_prob_std=marginal_prob_std_fn,
                                diffusion_coeff=diffusion_coeff_fn)

    train_loss = train(train_blob['data'], score_model, loss, optimizer, n_epochs, 
                        batch_size, device, num_workers, verbose=verbose, lrs=lrs)

    eval_data = eval(
        train_blob['data'],
        reference_blob['data'],
        [e['data'] for e in eval_blobs],
        score_model, ode_likelihood, batch_size//2, device, num_workers,
        batches=batches, verbose=verbose)
    eval_data['train_loss'] = train_loss
    eval_data['feat_dim'] = feat_dim    
    eval_data['encoder'] = encoder
    return eval_data


def ask_tell_optuna(objective_func, study_name, storage_name, device):
    study = optuna.create_study(directions=['maximize', 'minimize'], study_name=study_name, storage=storage_name, load_if_exists=True)
    trial = study.ask()
    res = objective_func(trial, device)
    study.tell(trial, res)


def get_opuna_value(name, opt_values, trial):
    data_type,*values = opt_values
    if data_type == "int":
        min_value, max_value, step_scale = values
        if step_scale == "log":
            return trial.suggest_int(name, min_value, max_value, log=True)
        elif step_scale.startswith("uniform_"):
            step = int(step_scale.split("_")[1])
            return trial.suggest_int(name, min_value, max_value, step=step)
        else:
            return trial.suggest_int(name, min_value, max_value)
    elif data_type == "float":
        min_value, max_value, step_scale = values
        if step_scale == "log":
            return trial.suggest_float(name, min_value, max_value, log=True)
        elif step_scale.startswith("uniform_"):
            step = float(step_scale.split("_")[1])
            return trial.suggest_float(name, min_value, max_value, step=step)
        else:
            return trial.suggest_float(name, min_value, max_value)
    elif data_type == "categorical":
        return trial.suggest_categorical(name, values[0])
    else:
        raise ValueError(f"Unknown data type {data_type}")
    
# activation@: [categorical, [relu, gelu, swish, leaky_relu]]
# normalization@: [categorical, [batch_norm, group_norm, instance_norm, layer_norm, local_response_norm, none]]
# blocks@: [category, [dense_add_embed, dense_concat_embed]]
ACTIVATIONS = {
    'relu': torch.nn.ReLU(),
    'gelu': torch.nn.GELU(),
    'silu': torch.nn.SiLU(),
    'leaky_relu': torch.nn.LeakyReLU(),
    }
NORMALIZATIONS = {
    'batch_norm': torch.nn.BatchNorm1d,
    'instance_norm': torch.nn.InstanceNorm1d,
    'layer_norm': torch.nn.LayerNorm,
    'local_response_norm': torch.nn.LocalResponseNorm,
    'none': torch.nn.Identity,
    }
BLOCKS = {
    'dense_add_embed': DenseAddEmbed,
    }

def str_to_torch(name, value):
    if name == 'activation':
        return ACTIVATIONS[value]
    elif name == 'normalization':
        return NORMALIZATIONS[value]
    elif name == 'blocks':
        return BLOCKS[value]
    else:
        return value

def objective(config, trial, device):
    cfg = copy.deepcopy(config)
    cfg['id'] = trial.number
    for name, value in config.items():
        if "@" in name:
            name = name.replace("@", "")
            value = get_opuna_value(name, value, trial)
        obj = str_to_torch(name, value)
        cfg[name] = obj
    cfg['device'] = device
    print(f"Running trial {trial.number}, device {cfg['device']}")
    eval_data = train_eval(cfg)

    score = np.mean(eval_data['score'])
    score_ref = np.mean(eval_data['score_ref'])
    loss = eval_data['train_loss'][-1]
    auc = float(np.mean(eval_data['auc']))
    trial.set_user_attr('auc', auc)
    trial.set_user_attr('frp',float(np.mean(eval_data['fpr'])))
    trial.set_user_attr('ref_auc', float(eval_data['ref_auc']))
    trial.set_user_attr('ref_fpr', float(eval_data['ref_fpr']))
    trial.set_user_attr('score', float(score))
    trial.set_user_attr('score_ref', float(score_ref))
    trial.set_user_attr('loss', float(loss))
    trial.set_user_attr('feat_dim', int(eval_data['feat_dim']))
    trial.set_user_attr('encoder', eval_data['encoder'])
    trial.set_user_attr('sigma', int(cfg['sigma']))
    trial.set_user_attr('lr', float(cfg['lr']))
    ref_auc = float(np.abs(eval_data['ref_auc']-0.5))

    return auc, ref_auc

def main_optuna():
    parser= argparse.ArgumentParser()
    parser.add_argument('config', help='Path to config file')
    args = parser.parse_args()
    path = args.config
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cfg = copy.deepcopy(config)
    jobs = []
    storage_name = config['db']
    print(f"Use: optuna-dashboard {storage_name}")
    names = []
    values = []
    for name, value in config.items():
        if "$" in name:
            name = name.replace("$", "")
            random.shuffle(value)
            names.append(name)
            values.append(value)
    
    for vs in itertools.product(*values):
        study_name = f'{cfg["name"]}'
        for i, n in enumerate(names):
            cfg[n] = vs[i]
            study_name += f'_{n}_{vs[i]}'
        trials = cfg['trials']
        partial_objective = functools.partial(objective, copy.deepcopy(cfg))
        jobs += [(partial_objective, study_name, storage_name) for _ in range(trials)]

    gpu_nodes = cfg['nodes']*cfg['jobs_per_node']
    random.shuffle(jobs)
    parallelize(ask_tell_optuna, jobs, gpu_nodes)

def main_config():
    parser= argparse.ArgumentParser()
    parser.add_argument('config', help='Path to config file')
    args = parser.parse_args()
    path = args.config
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cfg = copy.deepcopy(config)
    for name, value in config.items():
        value = str_to_torch(name, value)
        cfg[name] = value
    eval_data = train_eval(cfg)
    results = np.mean(eval_data['auc'])
    return results
            
