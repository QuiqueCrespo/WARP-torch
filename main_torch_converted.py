"""
Main training script for PyTorch models - Converted from JAX
"""

#%% Import the necessary modules
from utils import *
from loaders import *
from models_torch import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_theme(context='poster',
             style='ticks',
             font='sans-serif',
             font_scale=1,
             color_codes=True,
             rc={"lines.linewidth": 1})
mpl.rcParams['savefig.facecolor'] = 'w'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['savefig.bbox'] = 'tight'

from IPython.display import display, Math
import yaml
import argparse
import os
import time
import sys
import pickle

import warnings
warnings.filterwarnings("ignore")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

#%% Parse the command line arguments or use the default config.yaml file

try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False

if _in_ipython_session:
    args = argparse.Namespace(config_file='config.yaml')
    print("Notebook session: Using default config.yaml file")
else:
    if len(sys.argv) == 1:
        args = argparse.Namespace(config_file='config.yaml')
        print("No config file provided: Using default config.yaml file")
    elif len(sys.argv) > 1:
        args = argparse.Namespace(config_file=sys.argv[1])
        print(f"Using command line {sys.argv[1]} as config file")
    else:
        print("Usage: python main_torch_converted.py <config_file>")
        sys.exit(1)

with open(args.config_file, 'r') as file:
    config = yaml.safe_load(file)

model_type = config['model']['model_type']
if model_type == "wsm":
    print("\n\n+=+=+=+=+ Training Weight Space Model +=+=+=+=+\n")
elif model_type == "gru":
    print("\n\n+=+=+=+=+ Training Gated Recurrent Unit Model +=+=+=+=+\n")
elif model_type == "lstm":
    print("\n\n+=+=+=+=+ Training Long Short Term Memory Model +=+=+=+=+\n")
elif model_type == "ffnn":
    print("\n\n+=+=+=+=+ Training Feed-Forward Neural Network Model +=+=+=+=+\n")
else:
    print("\n\n+=+=+=+=+ Training Unknown Model +=+=+=+=+\n")
    raise ValueError(f"Unknown model type: {model_type}")

seed = config['general']['seed']
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

#%% Setup the run folder and data folder
train = config['general']['train']
classification = config['general']['classification']
data_path = config['general']['data_path']
save_path = config['general']['save_path']

### Create and setup the run and data folders
if train:
    if save_path is not None:
        run_folder = save_path
    else:
        run_folder = make_run_folder(f'./runs/{config["general"]["dataset"]}/')
    data_folder = data_path
else:
    run_folder = "./"
    data_folder = f"../../../{data_path}"

print("Using run folder:", run_folder)
## Copy the module files to the run folder
logger, checkpoints_folder, plots_folder, artefacts_folder = setup_run_folder(run_folder, training=train)

## Copy the config file to the run folder, renaming it as config.yaml
if not os.path.exists(run_folder+"config.yaml"):
    test_config = config.copy()
    test_config['general']['train'] = False
    with open(run_folder+"config.yaml", 'w') as file:
        yaml.dump(test_config, file)

## Print the config file using the logger
logger.info(f"Config file: {args.config_file}")
logger.info("==== Config file's contents ====")

for key, value in config.items():
    if isinstance(value, dict):
        logger.info(f"{key}:")
        for sub_key, sub_value in value.items():
            logger.info(f"  {sub_key}: {sub_value}")
    else:
        logger.info(f"{key}: {value}")

#%% Create the data loaders and visualize a few samples

trainloader, validloader, testloader, data_props = make_dataloaders(data_folder, config)
nb_classes, seq_length, data_size, width = data_props

print("Total number training samples:", len(trainloader.dataset))

batch = next(iter(trainloader))
(in_sequence, times), output = batch
logger.info(f"Input sequence shape: {in_sequence.shape}")
logger.info(f"Labels/Output Sequence shape: {output.shape}")
logger.info(f"Seq length: {seq_length}")
logger.info(f"Data size: {data_size}")
logger.info(f"Min/Max in the dataset: {np.min(in_sequence), np.max(in_sequence)}")
logger.info("Number of batches:")
logger.info(f"  - Train: {trainloader.num_batches}")
logger.info(f"  - Valid: {validloader.num_batches}")
logger.info(f"  - Test: {testloader.num_batches}")

## Plot a few samples in a 4x4 grid
fig, axs = plt.subplots(4, 4, figsize=(10, 10), sharex=True)
colors = ['r', 'g', 'b', 'c', 'm', 'y']

dataset = config['general']['dataset']
image_datasets = ["mnist", "mnist_fashion", "cifar", "celeba", "pathfinder", "lra"]
dynamics_datasets = ["lorentz63", "lorentz96", "lotka", "trends", "mass_spring_damper", "cheetah", "electricity", "sine"]
repeat_datasets = ["lotka", "arc_agi", "icl", "traffic", "mitsui"]

res = (width, width, data_size)
dim0, dim1 = (-1, -1)
if dim1>= data_size:
    dim1 = 0
    logger.info(f"dim1 is out of bounds. Setting it to 0.")

for i in range(4):
    for j in range(4):
        idx = np.random.randint(0, in_sequence.shape[0])
        if dataset in image_datasets:
            to_plot = in_sequence[idx].reshape(res)[...,:3]
            if dataset=="celeba":
                to_plot = (to_plot + 1) / 2
            axs[i, j].imshow(to_plot, cmap='gray')
        elif dataset=="trends":
            axs[i, j].plot(in_sequence[idx], color=colors[output[idx]])
        elif dataset in repeat_datasets:
            axs[i, j].plot(output[idx, :, dim0], color=colors[(i*j)%len(colors)], linestyle='-', lw=1, alpha=0.5)
            axs[i, j].plot(output[idx, :, dim1], color=colors[(i*j)%len(colors)], linestyle='--', lw=1, alpha=0.5)
            axs[i, j].plot(in_sequence[idx, :, dim0], color=colors[(i*j)%len(colors)], lw=3)
            axs[i, j].plot(in_sequence[idx, :, dim1], color=colors[(i*j)%len(colors)], linestyle='--', lw=3)
        else:
            axs[i, j].plot(in_sequence[idx, :, dim0], color=colors[int(output[idx])%len(colors)], lw=3)
            axs[i, j].plot(in_sequence[idx, :, dim1], color=colors[int(output[idx])%len(colors)], linestyle='--', lw=3)

        if dataset not in repeat_datasets:
            axs[i, j].set_title(f"Class: {output[idx]}", fontsize=12)
        axs[i, j].axis('off')

plt.suptitle(f"{dataset.upper()} Training Samples", fontsize=20)
plt.draw()
plt.savefig(plots_folder+"samples_train.png", dpi=100, bbox_inches='tight')

# %% Define the model and loss function

if not classification:
    nb_classes = None
model = make_model(data_size, nb_classes, config, logger)
model = model.to(device)
untrained_model = pickle.loads(pickle.dumps(model))  # Deep copy

nb_recons_loss_steps = config['training']['nb_recons_loss_steps']
use_nll_loss = config['training']['use_nll_loss']

def loss_fn(model, batch):
    """
    Loss function for the model.
    Args:
        model: PyTorch model
        batch: ((X_true, times), labels_or_outputs)
    Returns:
        loss: scalar loss value
        aux: auxiliary data (accuracy for classification, loss for regression)
    """

    if not classification:  ## Regression (forecasting) task
        (X_true, times), X_true_out = batch

        # Move to device
        X_true = torch.from_numpy(X_true).float().to(device)
        times = torch.from_numpy(times).float().to(device)
        X_true_out = torch.from_numpy(X_true_out).float().to(device)

        X_recons = model(X_true, times, inference_start=None)

        if nb_recons_loss_steps is not None:
            # Randomly sample steps in the sequence
            batch_size, nb_timesteps = X_true.shape[0], X_true.shape[1]
            indices_0 = torch.arange(batch_size, device=device)
            indices_1 = torch.randint(0, nb_timesteps, (batch_size, nb_recons_loss_steps), device=device)

            X_recons_ = torch.stack([X_recons[indices_0, indices_1[:,j]] for j in range(nb_recons_loss_steps)], dim=1)

            if dataset not in repeat_datasets:
                X_true_ = torch.stack([X_true[indices_0, indices_1[:,j]] for j in range(nb_recons_loss_steps)], dim=1)
            else:
                X_true_ = torch.stack([X_true_out[indices_0, indices_1[:,j]] for j in range(nb_recons_loss_steps)], dim=1)
        else:
            X_recons_ = X_recons
            if dataset not in repeat_datasets:
                X_true_ = X_true
            else:
                X_true_ = X_true_out

        if use_nll_loss:
            means = X_recons_[:, :, :data_size]
            stds = X_recons_[:, :, data_size:]
            loss_r = torch.log(stds) + 0.5*((X_true_ - means)/stds)**2
        else:
            loss_r = F.mse_loss(X_recons_, X_true_, reduction='none')

        loss = torch.mean(loss_r)
        return loss, loss

    else:  ## Classification task
        (X_true, times), Ys = batch

        # Move to device
        X_true = torch.from_numpy(X_true).float().to(device)
        times = torch.from_numpy(times).float().to(device)
        Ys = torch.from_numpy(Ys).long().to(device)

        Y_hat = model(X_true, times, inference_start=None)

        # Cross entropy loss
        loss = F.cross_entropy(Y_hat[:, -1], Ys)

        acc = (torch.argmax(Y_hat[:, -1], dim=-1) == Ys).float().mean()

        return loss, acc


def train_step(model, batch, optimizer):
    """Single training step."""
    model.train()
    optimizer.zero_grad()

    loss, aux = loss_fn(model, batch)
    loss.backward()

    # Gradient clipping
    if config['optimizer']['gradients_lim'] is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['gradients_lim'])

    optimizer.step()

    return loss.item(), aux if isinstance(aux, float) or isinstance(aux, int) else aux.item()


@torch.no_grad()
def forward_pass(model, X, times, inference_start=None):
    """Forward pass without gradients."""
    model.eval()
    X = torch.from_numpy(X).float().to(device)
    times = torch.from_numpy(times).float().to(device)

    X_recons = model(X, times, inference_start)
    return X_recons.cpu().numpy()


val_criterion = config['training']['val_criterion']
if val_criterion == "nll":
    if not config['training']['stochastic']:
        raise ValueError("NLL val loss can only be used if trained in stochastic mode.")
    elif "smooth_inference" in config['training'] and config['training']['smooth_inference']:
        raise ValueError("NLL val loss cannot be used in smooth inference mode.")

logger.info(f"Validation criterion is: {val_criterion}")


def eval_on_dataloader(model, dataloader, inference_start):
    """Evaluate the model on the designated set."""
    model.eval()
    cri_vals = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (X_true, times), X_labs_outs = batch

            if not classification:
                X_recons = forward_pass(model, X_true, times, inference_start=inference_start)

                if dataset in repeat_datasets:
                    X_gt = X_labs_outs
                else:
                    X_gt = X_true

                if use_nll_loss:
                    means_, stds_ = np.split(X_recons, 2, axis=-1)
                else:
                    means_ = X_recons

                if val_criterion == "mse":
                    loss_val = np.mean((means_ - X_gt)**2)
                elif val_criterion == "mae":
                    loss_val = np.mean(np.abs(means_ - X_gt))
                elif val_criterion == "rmse":
                    loss_val = np.sqrt(np.mean((means_ - X_gt)**2))
                elif val_criterion == "nll":
                    loss_val = np.log(stds_) + 0.5*((X_gt - means_)/stds_)**2
                    loss_val = np.mean(loss_val)
                elif val_criterion == "neg_mitsui":
                    loss_val = -mitsui_metric(means_, X_gt)
                else:
                    raise ValueError(f"Unknown validation criterion for regression: {val_criterion}")

            else:
                Y_hat = forward_pass(model, X_true, times, inference_start=inference_start)
                if val_criterion == "cce":
                    Y_hat_tensor = torch.from_numpy(Y_hat[:, -1]).float().to(device)
                    labs_tensor = torch.from_numpy(X_labs_outs).long().to(device)
                    loss_val = F.cross_entropy(Y_hat_tensor, labs_tensor).item()
                elif val_criterion == "error_rate":
                    acc = np.mean(np.argmax(Y_hat[:, -1], axis=-1) == X_labs_outs)
                    loss_val = 1 - acc
                elif val_criterion == "f1_score":
                    preds = np.argmax(Y_hat[:, -1], axis=-1)
                    f1_macro = f1_score_macro(y_true=X_labs_outs, y_pred=preds, nb_classes=nb_classes)
                    loss_val = 1 - f1_macro
                else:
                    raise ValueError(f"Unknown validation criterion for classification: {val_criterion}")

            cri_vals.append(loss_val)

    return np.mean(cri_vals), np.median(cri_vals), np.min(cri_vals)


#%% Train and validate the model

if train:
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['init_lr'],
        weight_decay=1e-5
    )

    # Setup learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['optimizer']['on_plateau']['factor'],
        patience=config['optimizer']['on_plateau']['patience'],
        threshold=config['optimizer']['on_plateau']['rtol'],
        cooldown=config['optimizer']['on_plateau']['cooldown'],
        min_lr=config['optimizer']['init_lr'] * config['optimizer']['on_plateau']['min_scale']
    )

    # Load model if exists
    if os.path.exists(artefacts_folder+"model.pt"):
        checkpoint = torch.load(artefacts_folder+"model.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model found in run folder. Finetuning from these.")
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            logger.info("No optimizer state for finetuning. Starting from scratch.")
    else:
        logger.info("No model found in run folder. Training from scratch.")

    losses = []
    med_losses_per_epoch = []
    lr_scales = []

    val_losses = []
    best_val_loss = np.inf
    best_val_loss_epoch = 0

    print_every = config['training']['print_every']
    save_every = config['training']['save_every']
    valid_every = config['training']['valid_every']
    inf_start = config["training"]["inference_start"]

    nb_epochs = config['training']['nb_epochs']
    logger.info(f"\n\n=== Beginning training ... ===")
    logger.info(f"  - Number of epochs: {nb_epochs}")
    logger.info(f"  - Number of batches: {trainloader.num_batches}")
    logger.info(f"  - Total number of GD steps: {trainloader.num_batches*nb_epochs}")

    start_time = time.time()

    for epoch in range(nb_epochs):
        epoch_start_time = time.time()
        losses_epoch = []
        aux_epoch = []

        for i, batch in enumerate(trainloader):
            loss, aux = train_step(model, batch, optimizer)

            losses_epoch.append(loss)
            losses.append(loss)
            aux_epoch.append(aux)

            lr_scales.append(optimizer.param_groups[0]['lr'])

        mean_epoch, median_epoch = np.mean(losses_epoch), np.median(losses_epoch)
        epoch_end_time = time.time() - epoch_start_time

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
            logger.info(
                f"Epoch {epoch:-4d}/{nb_epochs:-4d}     Train Loss   -Mean: {mean_epoch:.6f},   -Median: {median_epoch:.6f},   -Latest: {loss:.6f},     -WallTime: {epoch_end_time:.2f} secs"
            )

            if classification:
                logger.info(f"Average train classification accuracy: {np.mean(aux_epoch)*100:.2f}%")

        if epoch%save_every==0 or epoch==nb_epochs-1:
            if epoch==nb_epochs-1:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch
                }, checkpoints_folder+f"model_{epoch}.pt")

            np.save(artefacts_folder+"losses.npy", np.array(losses))
            np.save(artefacts_folder+"lr_scales.npy", np.array(lr_scales))
            np.savez(artefacts_folder+"val_losses.npz", losses=np.array(val_losses), best_epoch=best_val_loss_epoch, best_loss=best_val_loss)

            # Save best model based on training loss
            med_losses_per_epoch.append(median_epoch)
            if epoch==0 or median_epoch<=np.min(med_losses_per_epoch[:-1]):
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch
                }, artefacts_folder+"model_train.pt")
                logger.info("Best model on training set saved ...")

        if (valid_every is not None) and (epoch%valid_every==0) or (epoch==nb_epochs-1):
            val_mean_loss, val_median_loss, _ = eval_on_dataloader(model, validloader, inference_start=inf_start)
            val_losses.append(val_mean_loss)

            logger.info(
                f"Epoch {epoch:-4d}/{nb_epochs:-4d}     Validation Loss   +Mean: {val_mean_loss:.6f},   +Median: {val_median_loss:.6f}"
            )

            # Update scheduler
            scheduler.step(val_mean_loss)

            # Save best model based on validation loss
            if epoch==0 or val_mean_loss<=np.min(val_losses[:-1]):
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch
                }, artefacts_folder+"model.pt")
                logger.info("Best model on validation set saved ...")
                best_val_loss = val_mean_loss
                best_val_loss_epoch = epoch

        if epoch==3:  # Check GPU usage
            if torch.cuda.is_available():
                os.system("nvidia-smi")
                os.system("nvidia-smi >> "+artefacts_folder+"training.log")

    wall_time = time.time() - start_time
    logger.info("\nTraining complete. Total time: %d hours %d mins %d secs" %seconds_to_hours(wall_time))

    # Restore best model
    if os.path.exists(artefacts_folder+"model.pt"):
        checkpoint = torch.load(artefacts_folder+"model.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Best model from epoch {best_val_loss_epoch} restored.")
    elif os.path.exists(artefacts_folder+"model_train.pt"):
        checkpoint = torch.load(artefacts_folder+"model_train.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Best model on 'training set' restored.")
    else:
        logger.info("No 'best' model found. Using the last model.")

else:
    if os.path.exists(artefacts_folder+"model.pt"):
        checkpoint = torch.load(artefacts_folder+"model.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Best validation model restored.")
    elif os.path.exists(artefacts_folder+"model_train.pt"):
        checkpoint = torch.load(artefacts_folder+"model_train.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Best model on 'training set' restored.")
    else:
        raise ValueError("No model found to load. You might want to use one from a checkpoint.")

    try:
        losses = np.load(artefacts_folder+"losses.npy")
        lr_scales = np.load(artefacts_folder+"lr_scales.npy")
        val_losses_raw = np.load(artefacts_folder+"val_losses.npz")
        val_losses = val_losses_raw['losses']
        best_val_loss_epoch = val_losses_raw['best_epoch'].item()
        best_val_loss = val_losses_raw['best_loss'].item()
    except:
        losses = []
        val_losses = []

    logger.info(f"Model loaded from {run_folder}model.pt")

# %% Visualise the training (and validation) losses

if not os.path.exists(artefacts_folder+"losses.npy"):
    try:
        with open(artefacts_folder+"training.log", 'r') as f:
            lines = f.readlines()
        losses = []
        search_term = "Train Loss (Mean)"
        for line in lines:
            if search_term in line:
                loss = float(line.split(f"{search_term}: ")[1].strip())
                losses.append(loss)
        logger.info("Losses found in the training.log file")
    except:
        logger.info("No losses found in the training.log file")

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 5))

clean_losses = np.array(losses)
train_steps = np.arange(len(losses))
if not classification:
    loss_name = "NLL" if use_nll_loss else r"$L_2$"
else:
    loss_name = "Cross-Entropy"
ax = sbplot(train_steps, clean_losses, color="purple", title="Loss History", x_label='Train Steps', y_label=loss_name, ax=ax, y_scale="linear" if use_nll_loss else "log", label="Train")
ax.legend(fontsize=16, loc='upper left')
ax.yaxis.label.set_color('purple')

# Twin axis for validation losses
nb_epochs = config['training']['nb_epochs']
valid_every = config['training']['valid_every']
if len(val_losses) > 0:
    val_col = "teal"
    ax_ = ax.twinx()
    epochs_ids = (np.arange(0, nb_epochs, valid_every).tolist() + [nb_epochs-1])[:len(val_losses)]
    val_steps_ids = (np.array(epochs_ids)+1) * trainloader.num_batches
    ax_ = sbplot(val_steps_ids, val_losses, ".-", color=val_col, label=f"Valid", y_label=f'{val_criterion.upper()}', ax=ax_, y_scale="linear" if val_criterion in ["nll", "cce", "error_rate"] else "log", linewidth=3)
    ax_.legend(fontsize=16, loc='upper right')
    ax_.yaxis.label.set_color(val_col)

clean_losses = np.where(clean_losses<np.percentile(clean_losses, 96), clean_losses, np.nan)
ax2 = sbplot(train_steps, clean_losses, title="Loss History (96th Percentile)", x_label='Train Steps', y_label=loss_name, ax=ax2, y_scale="linear" if use_nll_loss else "log")

plt.draw()
plt.savefig(plots_folder+"loss.png", dpi=100, bbox_inches='tight')

if os.path.exists(artefacts_folder+"lr_scales.npy"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax = sbplot(lr_scales, "g-", title="LR Scales", x_label='Train Steps', ax=ax, y_scale="log")
    plt.draw()
    plt.savefig(plots_folder+"lr_scales.png", dpi=100, bbox_inches='tight')

# Plot validation losses in detail
nb_epochs = config['training']['nb_epochs']
valid_every = config['training']['valid_every']
val_ids = (np.arange(0, nb_epochs, valid_every).tolist() + [nb_epochs-1])[:len(val_losses)]
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax = sbplot(val_ids, val_losses, title=f"{val_criterion.upper()} on Valid Set at Various Epochs", x_label='Epoch', y_label=f'{val_criterion}', ax=ax, y_scale="log", linewidth=3)
plt.axvline(x=best_val_loss_epoch, color='r', linestyle='--', linewidth=3, label=f"Best {val_criterion.upper()}: {best_val_loss:.6f} at Epoch {best_val_loss_epoch}")
plt.legend(fontsize=16)
plt.draw()
plt.savefig(plots_folder+f"checkpoints_{val_criterion.lower()}.png", dpi=100, bbox_inches='tight')
logger.info(f"Best model found at epoch {best_val_loss_epoch} with {val_criterion}: {best_val_loss:.6f}")

# %% Other visualisations of the model

if config["model"]["model_type"] == "wsm":
    # Visualise the distribution of values along the main diagonal of A
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    A_diag = torch.diag(model.As[0]).cpu().numpy()
    axs[0].hist(A_diag, bins=100)
    axs[0].set_title("Histogram of diagonal values of A")

    if hasattr(model.thetas_init[0], 'weight'):
        theta_trained = model.thetas_init[0].weight.data.cpu().numpy().flatten()
        theta_untrained = untrained_model.thetas_init[0].weight.data.cpu().numpy().flatten()
        axs[1].hist(theta_trained, bins=100, label="After Training")
        axs[1].hist(theta_untrained, bins=100, alpha=0.5, label="Before Training", color='r')
        axs[1].set_title(r"Histogram of $\theta_0$ values")
        plt.legend()

    plt.draw()
    plt.savefig(plots_folder+"A_theta_histograms.png", dpi=100, bbox_inches='tight')

    # Plot B values
    if isinstance(model.Bs[0], nn.Parameter):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        B_vals = model.Bs[0].cpu().detach().numpy().flatten()
        ax.plot(B_vals, label="Values of B")
        ax.set_title("Values of B")
        ax.set_xlabel("Dimension")
        plt.draw()
        plt.savefig(plots_folder+"B_values.png", dpi=100, bbox_inches='tight')

    # Visualize A matrices
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    min_val = -0.0000
    max_val = 0.00003

    A_untrained = untrained_model.As[0].cpu().numpy()
    A_trained = model.As[0].cpu().detach().numpy()

    img = axs[0].imshow(A_untrained, cmap='viridis', vmin=min_val, vmax=max_val)
    axs[0].set_title("Untrained A")
    plt.colorbar(img, ax=axs[0], shrink=0.7)

    img = axs[1].imshow(A_trained, cmap='viridis', vmin=min_val, vmax=max_val)
    axs[1].set_title("Trained A")
    plt.colorbar(img, ax=axs[1], shrink=0.7)
    plt.draw()
    plt.savefig(plots_folder+"A_matrices.png", dpi=100, bbox_inches='tight')

logger.info("\n\n🎉 Script completed successfully!")
logger.info(f"Results saved in: {run_folder}")
