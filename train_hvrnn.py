import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from hnn_model import HNN
from Library.hgn_source.environments.environment import Environment, visualize_rollout
from Library.hgn_source.environments.datasets import EnvironmentSampler, EnvironmentLoader
from Library.hgn_source.environments.pendulum import Pendulum
from Library.hgn_source.environments.spring import Spring
from helper_functions import *
from Library.hgn_source.environments.gravity import NObjectGravity

class HVRNN(nn.Module):
    """
    The Hamiltonian Variational Recurrent Neural Network (HVRNN).

    """
    def __init__(self):
        super().__init__()

        # Transform input into q and p
        self.pre_processor = nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=16 * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.encoder_mean = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.encoder_logvar = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Prior
        self.prior = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.prior_mean = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.prior_logvar = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Decoder
        self.decoder = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        # HNN
        self.hnn = HNN()

    def encode(self, x):
        """
        Concatenates the inputs and passes them through the encoder network.

        Args:
            x (Tensor): Tensor of shape [batch_size, 16, height, width] representing the input data.


        Returns:
            mean (Tensor): Tensor of shape [batch_size, 16, height, width] representing the mean of the encoder distribution.
            logvar (Tensor): Tensor of shape [batch_size, 16, height, width] representing the log variance of the encoder distribution.

        """
        # print("Encoder In", x.shape)
        enc_x = self.encoder(x)
        # print("Encoder Out", enc_x.shape)

        mean = self.encoder_mean(enc_x)
        logvar = self.encoder_logvar(enc_x)

        return mean, logvar

    def reparameterisation(self, mean, logvar):
        """
        Applies the reparameterization trick to the inputs.

        Args:
            mean (Tensor): Tensor of shape [batch_size, 16, height, width] representing the mean of the encoder distribution.
            logvar (Tensor): Tensor of shape [batch_size, 16, height, width] representing the log variance of the encoder distribution.


        Returns:
            A latent encoding of the input sequence, where z = mean + logvar * eps.

        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mean + logvar * eps

    def pre_processor_fn(self, x):
        """
        Passes the input through the pre-processor network.

        NOTE: This expects in_channels = seq_len * channels. MUST change in above network each time as it is hard coded.

        Args:
            x (Tensor): Tensor of shape [batch_size, seq_len*channels, height, width] representing the input data.

        Returns:
            q (Tensor): Tensor of shape [batch_size, 16, height, width] representing the position.
            p (Tensor): Tensor of shape [batch_size, 16, height, width] representing the momentum.

        """
        x = self.pre_processor(x)

        half_len = int(x.shape[1] / 2)
        q = x[:, :half_len]
        p = x[:, half_len:]

        return q, p

    def get_prior(self, x):
        """
        Passes the input through the prior network and returns the parameters of the distribution.

        Args:
            x (Tensor): Tensor of shape [batch_size, 16, height, width] representing the input data.

        Returns:
            mean (Tensor): Tensor of shape [batch_size, 16, height, width] representing the mean of the prior distribution.
            logvar (Tensor): Tensor of shape [batch_size, 16, height, width] representing the log variance of the prior distribution.

        """
        # print("Pre Prior", x.shape) # [1, 32, 5, 5]
        prior_z = self.prior(x)
        # print("Prior encode:", prior_z.shape) # [1, 32, 5, 5]

        # Encode into mean and logvar
        mean = self.prior_mean(prior_z)
        logvar = self.prior_logvar(prior_z)

        return mean, logvar

    def decode(self, x):
        """
        Concatenates the input and passes it through the decoder network.

        Args:
            x (Tensor): Tensor of shape [batch_size, 16, height, width] representing the input data.

        Returns:
            dec_x (Tensor): Tensor of shape [batch_size, 3, height, width] representing the batch of output images.

        """

        dec_x = self.decoder(x)

        return dec_x

    def forward(self, x, target, steps=27):
        """
        Forward pass of the HVRNN.

        Args:
            x (Tensor): Tensor of shape [batch_size, seq_len, channels, height, width] representing the input frames.
            target (Tensor): Tensor of shape [batch_size, seq_len, channels, height, width] representing the target frames.
            steps (Integer): How many frames to loop over.

        Returns:
            output (Tensor): Tensor representing the predicted frames.
            full_loss (Tensor): Represents the training loss. To be back-propagated.
            forecast_loss (List): List of the frame-by-frame reconstruction loss.
            q_initial (Tensor): Tensor representing the initial position tensor. For use if sampling the prior.

        """

        batch_loss = 0
        forecast_loss = []
        output = []
        comb_output = torch.empty(torch.Size(list(target.shape))).to(device)

        # Pre-process input frames
        x = concatenate_input(x)
        # print("Converted Input", x.shape)

        q_next, p_next = self.pre_processor_fn(x)
        # print("Pre-Processor Out", q_next.shape, p_next.shape)
        q_initial = q_next.detach()

        for frame in range(steps):
            # print(frame)
            target_frame = target[:, frame, :]
            # print("Target Frame", target_frame.shape)

            z_mean, z_logvar = self.encode(p_next)

            # print("z_mean, z_logvar", z_mean.shape, z_logvar.shape)

            prior_mean, prior_logvar = self.get_prior(q_next)
            # print("prior_mean, prior_logvar", prior_mean.shape, prior_logvar.shape)

            p_next = self.reparameterisation(z_mean, z_logvar)
            # print("Encoder out", p_next.shape)

            # Predict next q, p
            q_next, p_next = self.hnn.euler_step(q_next, p_next, self.hnn)
            # print("Hamiltonian Loop Out", q_next.shape, p_next.shape)

            dec_x = self.decode(q_next)
            # print("Decoder Out:", dec_x.shape)
            output.append(dec_x)
            comb_output[:, frame] = dec_x

            # Frame loss
            frame_loss = recon_loss_fn(x=target_frame, x_reconstructed=dec_x)
            # print("Frame Loss", frame_loss.item())
            batch_loss += frame_loss

            forecast_loss.append(frame_loss.item())

        double_kld_per_sample = double_kld_loss_fn(enc_mean=z_mean, enc_logvar=z_logvar, prior_mean=prior_mean, prior_logvar=prior_logvar)
        # print("Compare KLD", kld_per_sample, double_kld_per_sample)
        kld_out = kld_normalize(kld=double_kld_per_sample, prediction=comb_output)

        full_loss = kld_out + batch_loss
        # print("Full loss:", full_loss.item(), kld_out.item(), batch_loss.item())

        output = comb_output[0]

        return output, full_loss, forecast_loss, q_initial

    def sample(self, n_steps, q_0):
        """
        Generates rollout using samples from the prior instead of the encoder.

        Args:
            n_steps (Integer): Number of time steps to perform rollout.
            q_0 (Tensor): Tensor representing the initial position tensor.

        Returns:
            sample_output (List): List of decoded images.
            sample_comb[0] (Tensor): Tensor representing a batch of predicted frames.

        """
        sample_output = []
        sample_comb = torch.empty((2, n_steps, 3, 32, 32)).to(device)

        q_next = q_0.requires_grad_()

        for frame in range(n_steps):

            prior_mean, prior_logvar = self.get_prior(q_next)
            # print("prior_mean, prior_logvar", prior_mean.shape, prior_logvar.shape)

            p_next = self.reparameterisation(prior_mean, prior_logvar)
            # print("Encoder out", p_next.shape) # [2, 48, 4, 4]

            # Predict next q, p
            q_next, _ = self.hnn.euler_step(q_next, p_next, self.hnn)
            # print("Hamiltonian Loop Out", q_next.shape, p_next.shape)

            dec_x = self.decode(q_next)
            # print("Decoder Out:", dec_x.shape)
            sample_output.append(dec_x)
            sample_comb[:, frame] = dec_x

        return sample_output, sample_comb[0]


def train_hvrnn():
    """
    Function to train the hvrnn model.
    """

    # Create dataset
    trainloader, testloader = get_pendulum_data()

    torch.cuda.empty_cache()
    print("Loading model")

    # Initialise model

    model = HVRNN().to(device)
    model.train()
    #
    # # Define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Training loop
    outputs = dict()
    test_outputs = dict()
    p_outputs = dict()
    loss_tracker = []
    test_loss_tracker = []

    for epoch in range(10):
        running_loss = 0
        test_loss = 0
        kld_test_loss = 0
        recon_test_loss = 0

        # Iterate over DataLoader for training
        for id, batch in enumerate(trainloader):
            # image_outputs = []
            batch_loss = 0
            # # print("ID:", id)
            x_in = batch.to(device) # torch.squeeze(batch, dim=0).to(device)
            # print("Input X:", x.shape)

            # Split into input sequence and target
            total_length = x_in.shape[0]
            input_length = 5
            input_sequence = x_in[:, :input_length, :]
            target_sequence = x_in[:, input_length:, :]
            # print("Input Split:", total_length, input_sequence.shape, target_sequence.shape)

            # Pass through Model
            optimizer.zero_grad()

            output, batch_loss, forecast_loss, q_initial = model(input_sequence, target_sequence)
            # loss = loss_fn(x_out, input_sequence)

            batch_loss.backward()

            ###################################################
            # Code to check gradients
            ###################################################
            # print("Weights")
            # for name, param in model.named_parameters():
            #     try:
            #         print(name, param.grad.norm())
            #     except:
            #         print(name)

            ###################################################
            # Clip gradients if getting nan results
            ###################################################
            # max_norm = 1.0
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            ###################################################
            ###################################################


            optimizer.step()

            # print("Batch Loss", batch_loss.item())
            running_loss += batch_loss.item()

            ###################################################
            # Code to sample from prior
            ###################################################
            # if id == len(trainloader)-1 and epoch == 9:
            #     sample_output, sample_comb = model.sample(n_steps=27, q_0=q_initial)
            #     sample_comb = sample_comb.permute(0, 2, 3, 1).cpu().detach().numpy()
            #     visualize_rollout(sample_comb)
            #     get_frames(sequence=sample_comb, n_steps=27)
            ###################################################
            ###################################################


        ##############
        # Test
        ##############

        model.eval()
        for id, test_batch in enumerate(testloader):
            x_in_test = torch.squeeze(test_batch, dim=0).to(device)

            # Split into input sequence and target
            total_length_test = x_in_test.shape[0]
            input_length_test = 5
            input_sequence_test = x_in_test[:, :input_length_test, :]
            target_sequence_test = x_in_test[:, input_length_test:, :]
            # print("Input Split:", total_length_test, input_sequence_test.shape, target_sequence_test.shape)

            # Position and Momentum Split

            # Pass through Model
            output_test, batch_loss_test, forecast_loss_test, q_initial_test = model(input_sequence_test, target_sequence_test)
            # loss_test = loss_fn(x_out_test, input_sequence_test)

            test_loss += batch_loss_test.item()

            ###################################################
            # Code to sample from prior
            ###################################################
            # if id == len(testloader) - 1 and epoch == 9:
            #     sample_output_test, sample_comb_test = model.sample(n_steps=45, q_0=q_initial_test)
            #     sample_comb_test = sample_comb_test.permute(0, 2, 3, 1).cpu().detach().numpy()
            #     visualize_rollout(sample_comb_test)
            #     get_frames(sequence=sample_comb_test, n_steps=27)
            ###################################################
            ###################################################

        print("\tEpoch", epoch + 1, "complete!", "\tAverage Train Loss: ", running_loss / 100, "\tTest Loss: ", test_loss)

    target_sequence = target_sequence[0].permute(0, 2, 3, 1).cpu().detach().numpy()
    output = output.permute(0, 2, 3, 1).cpu().detach().numpy()

    target_sequence_test = target_sequence_test[0].permute(0, 2, 3, 1).cpu().detach().numpy()
    output_test = output_test.permute(0, 2, 3, 1).cpu().detach().numpy()

    visualize_rollout(target_sequence)
    visualize_rollout(output)

    visualize_rollout(target_sequence_test)
    # visualize_test(forecast_loss_test)
    visualize_rollout(output_test)

    # print(forecast_loss)
    # print(forecast_loss_test)
    #
    # visualize_test(forecast_loss)
    # visualize_test(forecast_loss_test)
    #
    # get_frames(sequence=target_sequence)
    # get_frames(sequence=output)
    # get_frames(sequence=target_sequence_test)
    # get_frames(sequence=output_test)

    return

def train_hvrnn_impute():
    """
    Function to train the hvrnn model for imputed data.
    """

    # Create dataset
    trainloader, testloader = get_pendulum_data()

    torch.cuda.empty_cache()
    print("Loading model")

    # Initialise model

    model = HVRNN().to(device)
    model.train()
    #
    # # Define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Training loop
    outputs = dict()
    test_outputs = dict()
    p_outputs = dict()
    loss_tracker = []
    test_loss_tracker = []

    for epoch in range(10):
        running_loss = 0
        test_loss = 0
        kld_test_loss = 0
        recon_test_loss = 0

        # Iterate over DataLoader for training
        for id, batch in enumerate(trainloader):
            # image_outputs = []
            batch_loss = 0
            # # print("ID:", id)
            x_in = batch.to(device) # torch.squeeze(batch, dim=0).to(device)
            # print("Input X:", x_in.shape)

            x_in_imputed = make_imputed_data(x_in.permute(0, 1, 3, 4, 2))
            # print(x_in_imputed.shape)
            x_in_imputed = x_in_imputed.permute(0, 1, 4, 2, 3)
            # print(x_in_imputed.shape)

            # Split into input sequence and target
            total_length = x_in.shape[0]
            input_length = 10
            input_sequence = x_in_imputed[:, :input_length, :]
            target_sequence = x_in
            # print("Input Split:", total_length, input_sequence.shape, target_sequence.shape)

            # Pass through Model
            optimizer.zero_grad()

            output, batch_loss, forecast_loss, q_initial = model(input_sequence, target_sequence)
            # loss = loss_fn(x_out, input_sequence)

            batch_loss.backward()

            ###################################################
            # Code to check gradients
            ###################################################
            # print("Weights")
            # for name, param in model.named_parameters():
            #     try:
            #         print(name, param.grad.norm())
            #     except:
            #         print(name)

            ###################################################
            # Clip gradients if getting nan results
            ###################################################
            # max_norm = 1.0
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            ###################################################
            ###################################################


            optimizer.step()

            # print("Batch Loss", batch_loss.item())
            running_loss += batch_loss.item()

            ###################################################
            # Code to sample from prior
            ###################################################
            # if id == len(trainloader)-1 and epoch == 9:
            #     sample_output, sample_comb = model.sample(n_steps=27, q_0=q_initial)
            #     sample_comb = sample_comb.permute(0, 2, 3, 1).cpu().detach().numpy()
            #     visualize_rollout(sample_comb)
            #     get_frames(sequence=sample_comb, n_steps=27)
            ###################################################
            ###################################################


        ##############
        # Test

        model.eval()
        for id, test_batch in enumerate(testloader):
            x_in_test = torch.squeeze(test_batch, dim=0).to(device)

            x_in_imputed_test = make_imputed_data(x_in_test.permute(0, 1, 3, 4, 2))
            # print(x_in_imputed.shape)
            x_in_imputed_test = x_in_imputed_test.permute(0, 1, 4, 2, 3)

            # Split into input sequence and target
            total_length_test = x_in_test.shape[0]  # 32
            input_length_test = 10
            input_sequence_test = x_in_imputed_test[:, :input_length_test, :]
            target_sequence_test = x_in_test # [12, 1, 32, 32]
            # print("Input Split:", total_length_test, input_sequence_test.shape, target_sequence_test.shape)

            # Position and Momentum Split

            # Pass through Model
            output_test, batch_loss_test, forecast_loss_test, q_initial_test = model(input_sequence_test, target_sequence_test)
            # loss_test = loss_fn(x_out_test, input_sequence_test)

            test_loss += batch_loss_test.item()

            ###################################################
            # Code to sample from prior
            ###################################################
            # if id == len(testloader) - 1 and epoch == 9:
            #     sample_output_test, sample_comb_test = model.sample(n_steps=45, q_0=q_initial_test)
            #     sample_comb_test = sample_comb_test.permute(0, 2, 3, 1).cpu().detach().numpy()
            #     visualize_rollout(sample_comb_test)
            #     get_frames(sequence=sample_comb_test, n_steps=27)
            ###################################################
            ###################################################

        print("\tEpoch", epoch + 1, "complete!", "\tAverage Train Loss: ", running_loss / 100, "\tTest Loss: ", test_loss)

    target_sequence = x_in[0].permute(0, 2, 3, 1).cpu().detach().numpy()
    output = output.permute(0, 2, 3, 1).cpu().detach().numpy()

    target_sequence_test = x_in_test[0].permute(0, 2, 3, 1).cpu().detach().numpy()
    output_test = output_test.permute(0, 2, 3, 1).cpu().detach().numpy()

    visualize_rollout(target_sequence)
    visualize_rollout(output)
    #
    visualize_rollout(target_sequence_test)
    # visualize_test(forecast_loss_test)
    visualize_rollout(output_test)
    #
    # print(forecast_loss)
    # print(forecast_loss_test)
    #
    # visualize_test(forecast_loss)
    # visualize_test(forecast_loss_test)
    #
    # get_frames(sequence=target_sequence)
    # get_frames(sequence=output)
    # get_frames(sequence=target_sequence_test)
    # get_frames(sequence=output_test)

    return

if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)

    GPU = True  # Choose whether to use GPU
    if GPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f'Using {device}')

    ##########################
    # Choose training function.
    # Note: Imputation require some changes to be made to some lines of code above as indicated.
    ##########################
    train_hvrnn() # Works

    # train_hvrnn_impute() # Works