import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from Library.hgn_source.environments.environment import Environment, visualize_rollout
from Library.hgn_source.environments.datasets import EnvironmentSampler, EnvironmentLoader
from Library.hgn_source.environments.pendulum import Pendulum
from Library.hgn_source.environments.spring import Spring
from Library.hgn_source.environments.gravity import NObjectGravity
from helper_functions import *

class VRNN(nn.Module):
    """
    The Variational Recurrent Neural Network. Re-implemented based on the paper by Chung et al.

    Note: This re-implementation uses convolutional layers in order to handle the image sequences.

    """
    def __init__(self):
        """
        Initialises the layers of the VRNN.

        Returns:
            train_loader (Dataloader): Train dataloader.
            test_loader (Dataloader): Test dataloader.

        """
        super().__init__()

        # Feature extractor - X
        self.psi_x = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Feature extractor - Z
        self.psi_z = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.encoder_mean = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.encoder_logvar = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Softplus()
        )

        # Prior
        self.prior = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.prior_mean = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.prior_logvar = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Softplus()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, output_padding=1),
            nn.Sigmoid()
        )

        # Recurrence
        self.rnn = nn.GRUCell(48 * 8 * 8, 16 * 8 * 8)

    def encode(self, x, h_0):
        """
        Concatenates the inputs and passes them through the encoder network.

        Args:
            x (Tensor): Tensor of shape [batch_size, 16, height, width] representing the input data.
            h_0 (Tensor): Tensor of shape [batch_size, 16, height, width] representing the current hidden state.


        Returns:
            mean (Tensor): Tensor of shape [batch_size, 64, height, width] representing the mean of the encoder distribution.
            logvar (Tensor): Tensor of shape [batch_size, 64, height, width] representing the log variance of the encoder distribution.

        """

        # print("Encoder In", x.shape, h_0.shape)
        input = torch.cat((x, h_0), dim=1)
        enc_x = self.encoder(input)
        # print("Encoder Out", enc_x.shape)

        mean = self.encoder_mean(enc_x)
        logvar = self.encoder_logvar(enc_x)

        return mean, logvar

    def reparameterisation(self, mean, logvar):
        """
        Applies the reparameterization trick to the inputs.

        Args:
            mean (Tensor): Tensor of shape [batch_size, 64, height, width] representing the mean of the encoder distribution.
            logvar (Tensor): Tensor of shape [batch_size, 64, height, width] representing the log variance of the encoder distribution.


        Returns:
            A latent encoding of the input sequence, where z = mean + logvar * eps.

        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mean + logvar * eps

    def get_prior(self, x):
        """
        Passes the input through the prior network and returns the parameters of the distribution.

        Args:
            x (Tensor): Tensor of shape [batch_size, 16, height, width] representing the input data.

        Returns:
            mean (Tensor): Tensor of shape [batch_size, 64, height, width] representing the mean of the prior distribution.
            logvar (Tensor): Tensor of shape [batch_size, 64, height, width] representing the log variance of the prior distribution.

        """
        # print("Pre Prior", x.shape)
        prior_z = self.prior(x)
        # print("Prior encode:", prior_z.shape)

        # Encode into mean and logvar
        mean = self.prior_mean(prior_z)
        logvar = self.prior_logvar(prior_z)

        return mean, logvar

    def decode(self, h_0, z_psi):
        """
        Concatenates the input and passes it through the decoder network.

        Args:
            h_0 (Tensor): Tensor of shape [batch_size, 16, height, width] representing the current hidden state.
            z_psi (Tensor): Tensor of shape [batch_size, 16, height, width] representing the output of the psi_z network.

        Returns:
            dec_x (Tensor): Tensor of shape [batch_size, 3, height, width] representing the batch of output images.

        """
        x = torch.cat((h_0, z_psi), dim=1)
        # print("Decoder Cat", x.shape)
        dec_x = self.decoder(x)

        return dec_x

    def gru_update(self, x_psi, z_psi, h_0):
        """
        Concatenates the inputs and passes them through the recurrent unit (GRU cell).

        Args:
            x_psi (Tensor): Tensor of shape [batch_size, 16, height, width] representing the output of the psi_x network.
            h_0 (Tensor): Tensor of shape [batch_size, 16, height, width] representing the current hidden state.
            z_psi (Tensor): Tensor of shape [batch_size, 16, height, width] representing the output of the psi_z network.

        Returns:
            processed_h_next (Tensor): Tensor of shape [batch_size, 16, height, width] representing the next value of the hidden state.

        """
        # print("GRU Input", x_psi.shape, z_psi.shape, h_0.shape)
        concat_input = torch.cat((x_psi, z_psi, h_0), dim=1)
        # print("Concat Input", concat_input.shape)

        processed_input = concat_input.reshape(-1, 48 * 8 * 8)
        # print("Processed Input", processed_input.shape)

        h_next = self.rnn(processed_input)
        # print("h next", h_next.shape)

        processed_h_next = h_next.reshape(-1, 16, 8, 8)
        # print("processed h next", processed_h_next.shape)

        return processed_h_next

    def forward(self, x, target=None, steps=27):
        """
        Forward pass of the VRNN.

        Args:
            x (Tensor): Tensor of shape [batch_size, seq_len, channels, height, width] representing the input frames.
            target (Tensor): Tensor of shape [batch_size, seq_len, channels, height, width] representing the target frames.
            steps (Integer): How many frames to loop over.

        Returns:
            output (Tensor): Tensor representing the predicted frames.
            full_loss (Tensor): Represents the training loss. To be back-propagated.
            forecast_loss (List): List of the frame-by-frame reconstruction loss.

        """
        n_steps = x.size(1) - 1  # Change to target.size(1) - 1 if doing imputation.

        batch_loss = 0
        forecast_loss = []
        output = []
        comb_output = torch.empty((2, n_steps, 3, 32, 32)).to(device)
        batch_loss = 0

        h_0 = torch.zeros((2, 16, 8, 8), dtype=torch.float32).to(device)
        # print("h_0", h_0.shape) # [2, 16, 8, 8]

        for frame in range(n_steps):
            if frame < 6:  # Use 10 if doing imputation
                # print("Using actual frame")
                current_frame = x[:, frame, :]
            else:
                # print("Using predicted frame")
                current_frame = dec_x

            target_frame = x[:, frame+1, :]  # Use target[:, frame+1, :] if doing imputation
            # print("Current Frame", current_frame.shape)

            processed_frame = self.psi_x(current_frame)
            # print("Processed Frame", processed_frame.shape)

            z_mean, z_logvar = self.encode(processed_frame, h_0)
            # print("z_mean, z_logvar", z_mean.shape, z_logvar.shape)

            prior_mean, prior_logvar = self.get_prior(h_0)
            # print("prior_mean, prior_logvar", prior_mean.shape, prior_logvar.shape)

            z = self.reparameterisation(z_mean, z_logvar)
            # print("Encoder out", z.shape)

            z_psi = self.psi_z(z)
            # print("Feature Extracted z", z_psi.shape)

            dec_x = self.decode(h_0, z_psi)
            # print("Decoder Out:", dec_x.shape)
            output.append(dec_x)
            comb_output[:, frame] = dec_x

            # Frame loss
            frame_loss = recon_loss_fn(x=target_frame, x_reconstructed=dec_x)
            # print("Frame Loss", frame_loss.item())
            batch_loss += frame_loss

            forecast_loss.append(frame_loss.item())

            h_0 = self.gru_update(processed_frame, z_psi, h_0)

        double_kld_per_sample = double_kld_loss_fn(enc_mean=z_mean, enc_logvar=z_logvar, prior_mean=prior_mean, prior_logvar=prior_logvar)
        # print("Compare KLD", kld_per_sample, double_kld_per_sample)
        kld_out = kld_normalize(kld=double_kld_per_sample, prediction=comb_output)

        full_loss = kld_out + batch_loss
        # print("Full loss:", full_loss.item(), kld_out.item(), batch_loss.item())

        output = comb_output[0]

        return output, full_loss, forecast_loss

    def sample(self, n_steps):
        """
        Generates rollout using samples from the prior instead of the encoder.

        Args:
            n_steps (Integer): Number of time steps to perform rollout.


        Returns:
            sample_output (List): List of decoded images.
            sample_comb[0] (Tensor): Tensor representing a batch of predicted frames.

        """
        sample_output = []
        sample_comb = torch.empty((2, n_steps, 3, 32, 32)).to(device)

        h_0 = torch.zeros((2, 16, 8, 8), dtype=torch.float32).to(device)
        # print("h_0", h_0.shape) # [2, 16, 8, 8]

        for frame in range(n_steps):

            prior_mean, prior_logvar = self.get_prior(h_0)
            # print("prior_mean, prior_logvar", prior_mean.shape, prior_logvar.shape)  # [2, 32, 8, 8]

            z = self.reparameterisation(prior_mean, prior_logvar)
            # print("Encoder out", z.shape) # [2, 64, 8, 8]

            z_psi = self.psi_z(z)
            # print("Feature Extracted z", z_psi.shape)  # [2, 32, 8, 8]

            dec_x = self.decode(h_0, z_psi)
            # print("Decoder Out:", dec_x.shape)  # [2, 3, 32, 32]
            sample_output.append(dec_x)
            sample_comb[:, frame] = dec_x

            processed_frame = self.psi_x(dec_x)

            h_0 = self.gru_update(processed_frame, z_psi, h_0)

        return sample_output, sample_comb[0]

def train_vrnn():
    """
    Function to train the vrnn model.
    """

    # Create dataset
    trainloader, testloader = get_pendulum_data()

    torch.cuda.empty_cache()
    print("Loading model")

    # Initialise model

    model = VRNN().to(device)
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

    for epoch in range(5):
        running_loss = 0
        test_loss = 0

        # Iterate over DataLoader for training
        for id, batch in enumerate(trainloader):
            # image_outputs = []
            batch_loss = 0
            # # print("ID:", id)
            x_in = batch.to(device) # torch.squeeze(batch, dim=0).to(device)
            # x_in = x_in[:, :27, :]
            # print("Input X:", x_in.shape) # [2, 27, 3, 32, 32]

            # Split into input sequence and target
            total_length = x_in.shape[0]  # 32
            input_length = 5
            input_sequence = x_in  # [:, :input_length, :]  # [2, 32, 3, 32, 32]
            target_sequence = x_in[:, 1:, :]  # [2, 31, 3, 32, 32]
            # print("Input Split:", total_length, input_sequence.shape, target_sequence.shape)

            # Pass through Model
            optimizer.zero_grad()

            output, batch_loss, forecast_loss = model(input_sequence)
            # loss = loss_fn(x_out, input_sequence)

            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()

            ##############################################
            # Code to sample from prior
            ##############################################
            # if id % 249 == 0 and epoch == 9:
            #     sample_output, sample_comb = model.sample(n_steps=20)
            #     sample_comb = sample_comb.permute(0, 2, 3, 1).cpu().detach().numpy()
            #     visualize_rollout(sample_comb)
            ##############################################



        ##############
        # Test
        ##############

        model.eval()
        for id, test_batch in enumerate(testloader):
            x_in_test = torch.squeeze(test_batch, dim=0).to(device)

            # Split into input sequence and target
            total_length_test = x_in_test.shape[0]  # 32
            input_length_test = 5
            input_sequence_test = x_in_test # [:, :input_length_test, :]  # [20, 1, 32, 32]
            target_sequence_test = x_in_test[:, 1:, :]  # [12, 1, 32, 32]
            # print("Input Split:", total_length_test, input_sequence_test.shape, target_sequence_test.shape)

            # Position and Momentum Split

            # Pass through Model
            output_test, batch_loss_test, forecast_loss_test = model(input_sequence_test)
            # loss_test = loss_fn(x_out_test, input_sequence_test)

            test_loss += batch_loss_test.item()

        print("\tEpoch", epoch + 1, "complete!", "\tAverage Train Loss: ", running_loss / 100, "\tTest Loss: ", test_loss)

    target_sequence = target_sequence[0].permute(0, 2, 3, 1).cpu().detach().numpy()
    output = output.permute(0, 2, 3, 1).cpu().detach().numpy()

    target_sequence_test = target_sequence_test[0].permute(0, 2, 3, 1).cpu().detach().numpy()
    output_test = output_test.permute(0, 2, 3, 1).cpu().detach().numpy()

    visualize_rollout(target_sequence)
    visualize_rollout(output)
    #
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

def train_vrnn_impute():
    """
    Function to train the vrnn model for imputed data.
    """

    # Create dataset
    trainloader, testloader = get_pendulum_data()

    torch.cuda.empty_cache()
    print("Loading model")

    # Initialise model

    model = VRNN().to(device)
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

    for epoch in range(5):
        running_loss = 0
        test_loss = 0

        # Iterate over DataLoader for training
        for id, batch in enumerate(trainloader):
            # image_outputs = []
            batch_loss = 0
            # # print("ID:", id)
            x_in = batch.to(device) # torch.squeeze(batch, dim=0).to(device)
            # x_in = x_in[:, :27, :]
            # print("Input X:", x_in.shape) # [2, 27, 3, 32, 32]

            x_in_imputed = make_imputed_data(x_in.permute(0, 1, 3, 4, 2))
            # print(x_in_imputed.shape)
            x_in_imputed = x_in_imputed.permute(0, 1, 4, 2, 3)
            # print(x_in_imputed.shape)

            # Split into input sequence and target
            total_length = x_in.shape[0]  # 32
            input_length = 10
            input_sequence = x_in_imputed[:, :input_length, :]  # [2, 10, 3, 32, 32]
            target_sequence = x_in  # [2, 31, 3, 32, 32]
            # print("Input Split:", total_length, input_sequence.shape, target_sequence.shape)

            # Pass through Model
            optimizer.zero_grad()

            output, batch_loss, forecast_loss = model(input_sequence, target_sequence)
            # loss = loss_fn(x_out, input_sequence)

            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()

            ##############################################
            # Code to sample from prior
            ##############################################
            # if id % 249 == 0 and epoch == 9:
            #     sample_output, sample_comb = model.sample(n_steps=20)
            #     sample_comb = sample_comb.permute(0, 2, 3, 1).cpu().detach().numpy()
            #     visualize_rollout(sample_comb)
            ##############################################



        ##############
        # Test
        ##############

        model.eval()
        for id, test_batch in enumerate(testloader):
            x_in_test = torch.squeeze(test_batch, dim=0).to(device)

            x_in_imputed_test = make_imputed_data(x_in_test.permute(0, 1, 3, 4, 2))
            # print(x_in_imputed.shape)
            x_in_imputed_test = x_in_imputed_test.permute(0, 1, 4, 2, 3)

            # Split into input sequence and target
            total_length_test = x_in_test.shape[0]  # 32
            input_length_test = 10
            input_sequence_test = x_in_imputed_test[:, :input_length_test, :]  # [20, 1, 32, 32]
            target_sequence_test = x_in_test # [12, 1, 32, 32]
            # print("Input Split:", total_length_test, input_sequence_test.shape, target_sequence_test.shape)

            # Position and Momentum Split

            # Pass through Model
            output_test, batch_loss_test, forecast_loss_test = model(input_sequence_test, target_sequence_test)
            # loss_test = loss_fn(x_out_test, input_sequence_test)

            test_loss += batch_loss_test.item()

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

    # visualize_test(forecast_loss)
    # visualize_test(forecast_loss_test)

    # get_frames(sequence=target_sequence, n_steps="every")
    # get_frames(sequence=output, n_steps="every")
    # get_frames(sequence=target_sequence_test, n_steps="every" )
    # get_frames(sequence=output_test, n_steps="every")

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
    train_vrnn()  # Works

    # train_vrnn_impute() # Works