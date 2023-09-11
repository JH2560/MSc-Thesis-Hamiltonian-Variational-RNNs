import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from Library.hgn_source.environments.datasets import EnvironmentSampler, EnvironmentLoader
from Library.hgn_source.environments.pendulum import Pendulum
from Library.hgn_source.environments.spring import Spring
from Library.hgn_source.environments.gravity import NObjectGravity

################################
# Data Loading and Processing
################################

def get_spring_data():
    """
    Returns train and test dataloaders for the Spring dataset as a tuple.
    Note: This function makes use of functions from the open-source HGN repository.

    Note 2: Setting "Seed" as "None" will generate random rollouts. To reproduce the test
    rollout used in the presentation (and in the report), set "Seed = 32" or "Seed=23".

    Returns:
        train_loader (Dataloader): Train dataloader.
        test_loader (Dataloader): Test dataloader.

    """

    sp = Spring(mass=.5, elastic_cst=2, damping_ratio=0.)

    train_dataset = EnvironmentSampler(environment=sp,
                                 dataset_len=10,
                                 number_of_frames=32,
                                 delta_time=.1,
                                 number_of_rollouts=2,
                                 img_size=32,
                                 color=True,
                                 noise_level=0.,
                                 radius_bound=(1.3, 2.3),
                                 seed=32)

    # Dataloader instance test, batch_mode disabled
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                        shuffle=False,
                                        batch_size=None)

    test_dataset = EnvironmentSampler(environment=sp,
                                 dataset_len=1,
                                 number_of_frames=32,
                                 delta_time=.1,
                                 number_of_rollouts=2,
                                 img_size=32,
                                 color=True,
                                 noise_level=0.,
                                 radius_bound=(1.3, 2.3),
                                 seed=32)

    # Dataloader instance test, batch_mode disabled
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                        shuffle=False,
                                        batch_size=None)

    return train_loader, test_loader

def get_pendulum_data():
    """
    Returns train and test dataloaders for the Pendulum dataset as a tuple.
    Note: This function makes use of functions from the open-source HGN repository.

    Note 2: Setting "Seed" as "None" will generate random rollouts. To reproduce the test
    rollout used in the presentation (and in the report), set "Seed = 32" or "Seed=23".

    Returns:
        train_loader (Dataloader): Train dataloader.
        test_loader (Dataloader): Test dataloader.

    """

    pd = Pendulum(mass=0.5, length=1, g=3)

    train_dataset = EnvironmentSampler(environment=pd,
                                 dataset_len=10,
                                 number_of_frames=32,
                                 delta_time=.1,
                                 number_of_rollouts=2,
                                 img_size=32,
                                 color=True,
                                 noise_level=0.,
                                 radius_bound=(1.3, 2.3),
                                 seed=32)

    # Dataloader instance test, batch_mode disabled
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                        shuffle=False,
                                        batch_size=None)

    test_dataset = EnvironmentSampler(environment=pd,
                                 dataset_len=1,
                                 number_of_frames=32,
                                 delta_time=.1,
                                 number_of_rollouts=2,
                                 img_size=32,
                                 color=True,
                                 noise_level=0.,
                                 radius_bound=(1.3, 2.3),
                                 seed=32)

    # Dataloader instance test, batch_mode disabled
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                        shuffle=False,
                                        batch_size=None)

    return train_loader, test_loader

def get_three_body_data():
    """
    Returns train and test dataloaders for the Three-body dataset as a tuple.
    Note: This function makes use of functions from the open-source HGN repository.

    Note 2: Setting "Seed" as "None" will generate random rollouts. To reproduce the test
    rollout used in the presentation (and in the report), set "Seed = 32" or "Seed=23".

    Returns:
        train_loader (Dataloader): Train dataloader.
        test_loader (Dataloader): Test dataloader.

    """

    og = NObjectGravity(mass=[1., 1., 1.], g=1., orbit_noise=0.05)

    train_dataset = EnvironmentSampler(environment=og,
                                 dataset_len=10,
                                 number_of_frames=32,
                                 delta_time=.1,
                                 number_of_rollouts=2,
                                 img_size=32,
                                 color=True,
                                 noise_level=0.,
                                 radius_bound=(1.3, 2.3),
                                 seed=32)

    # Dataloader instance test, batch_mode disabled
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                        shuffle=False,
                                        batch_size=None)

    test_dataset = EnvironmentSampler(environment=og,
                                 dataset_len=1,
                                 number_of_frames=32,
                                 delta_time=.1,
                                 number_of_rollouts=2,
                                 img_size=32,
                                 color=True,
                                 noise_level=0.,
                                 radius_bound=(1.3, 2.3),
                                 seed=32)

    # Dataloader instance test, batch_mode disabled
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                        shuffle=False,
                                        batch_size=None)

    return train_loader, test_loader

def concatenate_input(x):
    """
    Concatenate the seq_len and channels dimensions.

    The loaders from the HGN repository output tensors of [batch_size, seq_len, channels, height, width],
    in order to process them through our models, need to convert them to [batch_size, seq_len * channels, height, width].

    Args:
        x (Tensor): Tensor of shape [batch_size, seq_len, channels, height, width] representing the input data.


    Returns:
        reshaped_x (Tensor): Tensor of shape [batch_size, seq_len * channels, height, width] representing the reshaped input data.

    """

    # Obtain dimension sizes
    batch, seq_len, channels, h, w = x.size()

    # Concatenate seq_len and channels dimensions
    reshaped_x = x.reshape((batch, seq_len * channels, h, w))

    return reshaped_x

def make_imputed_data(input):
    """
    Replaces frames 4-8 of the input sequence with "Black" frames.

    Args:
        input (Tensor): Tensor of shape [batch_size, seq_len, height, width, channels] representing the input data.


    Returns:
        imputed_input (Tensor): Tensor of shape [batch_size, seq_len, height, width, channels] representing the imputed input data.

    """

    background_color = torch.Tensor([81./255, 88./255, 93./255])
    blank_frame = torch.ones((32, 32, 3))

    black_frame = torch.mul(blank_frame, background_color)

    imputed_input = input.detach()
    imputed_input[:, 3:6, :] = black_frame

    return imputed_input

################################
# Loss Functions
################################

def recon_loss_fn(x, x_reconstructed):
    """
    Returns the MSE between the predicted frames and the target frames.

    Args:
        x (tensor): Tensor of shape [Batch Size, Channels, Height, Width] representing the target frames.
        x_reconstructed (tensor): Tensor of shape [Batch Size, Channels, Height, Width] representing the predicted frames.

    Returns:
        recon_loss (Tensor) : Tensor representing the MSE loss.

    """

    loss_fn = nn.MSELoss()
    recon_loss = loss_fn(x_reconstructed, x)

    return recon_loss


def kld_loss_fn(mean, logvar):
    """
    Returns the Kullbach-Leibler Divergence (KLD) between the given distribution and a standard normal prior.

    Args:
        mean (Tensor): Tensor of shape [Batch Size, Channels, Height, Width] representing the mean of the given distribution.
        logvar (Tensor): Tensor of shape [Batch Size, Channels, Height, Width] representing the log variance of the given distribution.

    Returns:
        kld (Tensor) : Tensor representing the KLD loss

    """

    # Compute kld for each datapoint
    mean = mean.flatten(1)
    logvar = logvar.flatten(1)
    # print("Flattened Input", mean.shape, logvar.shape)

    kld_datapoint = -0.5 * torch.sum(logvar + 1 - mean.pow(2) - logvar.exp(), dim=1)
    # print("kld per datapoint", kld_datapoint.shape, kld_datapoint)

    # Return mean value over the batches
    kld = torch.mean(kld_datapoint, dim=0)
    # print("Mean KLD", kld.shape, kld)

    return kld


def double_kld_loss_fn(enc_mean, enc_logvar, prior_mean, prior_logvar):
    """
    Returns the Kullbach-Leibler Divergence (KLD) between the two given distributions.

    Args:
        enc_mean (Tensor): Tensor of shape [Batch Size, Channels, Height, Width] representing the mean of the encoder distribution.
        enc_logvar (Tensor): Tensor of shape [Batch Size, Channels, Height, Width] representing the log variance of the encoder distribution.
        prior_mean (Tensor): Tensor of shape [Batch Size, Channels, Height, Width] representing the mean of the prior distribution.
        prior_logvar (Tensor): Tensor of shape [Batch Size, Channels, Height, Width] representing the log variance of the prior distribution.

    Returns:
        kld (Tensor) : Tensor representing the KLD loss.

    """

    # Compute kld for each datapoint
    enc_mean = enc_mean.flatten(1)
    enc_logvar = enc_logvar.flatten(1)

    prior_mean = prior_mean.flatten(1)
    prior_logvar = prior_logvar.flatten(1)
    # print("Flattened Input", enc_mean.shape, enc_logvar.shape, prior_mean.shape, prior_logvar.shape)

    kld_component = (2 * (enc_logvar - prior_logvar)) + \
                    (prior_logvar.exp().pow(2) + (prior_mean - enc_mean).pow(2)) / enc_logvar.exp().pow(2) - 1

    kld_datapoint = 0.5 * torch.sum(kld_component, dim=1)
    # print("kld per datapoint", kld_datapoint.shape, kld_datapoint)

    # Return mean value over the batches
    kld = torch.mean(kld_datapoint, dim=0)
    # print("Mean KLD", kld.shape, kld)

    return kld


def kld_normalize(kld, prediction):
    """
    Returns the normalized Kullbach-Leibler Divergence (KLD). Normalized by no. of datapoints (i.e., no. of frames, channels and pixels of prediction frames).

    Args:
        kld (Tensor) : Tensor representing the KLD loss.
        prediction (Tensor): Tensor of shape [Batch Size, Channels, Height, Width] representing the predicted frames.

    Returns:
        kld_normalized (Tensor) : Tensor representing the normalized KLD loss

    """

    # Obtain number of datapoints in the prediction frames
    normalizer = prediction.flatten(1)
    # print("KLD Normal Flatten", normalizer.shape)

    num_datapoints = normalizer.size(1)
    # print("Number of Datapoints", num_datapoints)

    # Normalize kld
    kld_normalized = kld / num_datapoints
    # print("Normalized KLD", kld_normalized)

    return kld_normalized

def training_loss_fn(recon_loss, kld_loss, beta=1):
    """
    Returns the training loss, computed as: recon_loss + beta * kld_loss.

    Args:
        recon_loss (Tensor): Tensor representing the MSE loss.
        kld_loss (Tensor): Tensor representing the KLD loss.
        beta (Integer): Multiplier for the KLD loss.

    Returns:
        training_loss (Tensor) : Tensor representing the training loss.

    """

    training_loss = recon_loss + beta * kld_loss

    return training_loss

################################
# Visualisations
################################

def get_frames(sequence, n_steps = 27):
    """
    Returns a visualisation showing every nth frame of the input rollout sequence.

    Args:
        sequence (Tensor): Tensor  of shape [Seq_len, Height, Width, Channels] representing the rollout sequence to be visualised.
        n_steps (Integer): The number of steps between each frame capture.

    """

    if n_steps == 27:
        frames = [0, 4, 8, 12, 16, 20, 24, 26]
    elif n_steps == 45:
        frames = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]
    elif n_steps == "every":
        frames = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    # Obtain required frames
    display_item = []
    for frame in frames:
        display_item.append(sequence[frame])


    # Plot frames
    plt.figure(figsize=(10, 10))
    for i, x in enumerate(display_item):
        plt.subplot(1, len(frames)+1, i + 1)
        plt.axis("off")
        plt.title("Frame {}".format(frames[i]))
        plt.imshow(x)
    plt.show()

def visualize_test(test_losses):
    """
    Returns a visualisation showing plot of losses against time steps.

    Args:
        test_losses (List): A list of losses.

    """

    with torch.no_grad():
        x_vals = [i for i in range(len(test_losses))]
        plt.plot(x_vals, test_losses)
        plt.xlabel("Forecast Step")
        plt.ylabel("MSE Loss")
        plt.xlim(0, len(x_vals))
        plt.ylim(bottom=0)
        plt.show()