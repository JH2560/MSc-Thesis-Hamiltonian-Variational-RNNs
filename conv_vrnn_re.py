import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from Library.hgn_source.environments.environment import Environment, visualize_rollout
from Library.hgn_source.environments.datasets import EnvironmentSampler, EnvironmentLoader
from Library.hgn_source.environments.pendulum import Pendulum
from Library.hgn_source.environments.spring import Spring
from Library.hgn_source.environments.gravity import NObjectGravity
from helper_functions import get_frames

def get_data():
    """
    Obtains the raw data and splits into train and test splits.
    """

    # pd = Pendulum(mass=0.5, length=1, g=3)
    # sp = Spring(mass=.5, elastic_cst=2, damping_ratio=0.)
    og = NObjectGravity(mass=[1., 1., 1.], g=1., orbit_noise=0.05)

    trainDS = EnvironmentSampler(environment=og,
                                 dataset_len=300, # 1500 = 40 mins  # 5000 = 1.5hrs
                                 number_of_frames=32,
                                 delta_time=.1,
                                 number_of_rollouts=2,
                                 img_size=32,
                                 color=True,
                                 noise_level=0.,
                                 radius_bound=(1.3, 2.3),
                                 seed=32)  # Set as None = Random rollout each dataset. Note: Need at least 10000 to learn.

    # Dataloader instance test, batch_mode disabled
    train = torch.utils.data.DataLoader(trainDS,
                                        shuffle=False,
                                        batch_size=None)

    testDS = EnvironmentSampler(environment=og,
                                 dataset_len=1,
                                 number_of_frames=50,
                                 delta_time=.1,
                                 number_of_rollouts=2,
                                 img_size=32,
                                 color=True,
                                 noise_level=0.,
                                 radius_bound=(1.3, 2.3),
                                 seed=32)

    # Dataloader instance test, batch_mode disabled
    test = torch.utils.data.DataLoader(testDS,
                                        shuffle=False,
                                        batch_size=None)

    return train, test

class ModelB(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.input_dim = input_dim

        # feature-extracting transformations
        self.psi_x = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.psi_z = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # encoder
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

        # prior
        self.prior = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.prior_mean = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.prior_logvar = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Softplus()
        )

        # decoder
        self.decoder = nn.Sequential(
            # nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, output_padding=1),
            nn.Sigmoid()
        )

        # recurrence
        self.rnn = nn.GRUCell(48 * 8 * 8, 16 * 8 * 8)

    def concat_rgb(self, x):
        batch, seq_len, channels, h, w = x.size()

        return x.reshape((batch, seq_len * channels, h, w))

    def encode(self, x, h_0):
        # print("Encoder In", x.shape, h_0.shape) # [2, 16, 8, 8]
        input = torch.cat((x, h_0), dim=1)
        enc_x = self.encoder(input)
        # print("Encoder Out", enc_x.shape) # [2, 64, 8, 8]

        mean = self.encoder_mean(enc_x)
        logvar = self.encoder_logvar(enc_x)

        return mean, logvar

    def reparameterisation(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mean + logvar * eps

    def get_prior(self, x):
        # print("Pre Prior", x.shape) # [1, 32, 5, 5]
        prior_z = self.prior(x)
        # print("Prior encode:", prior_z.shape) # [1, 32, 5, 5]

        # Encode into mean and logvar
        mean = self.prior_mean(prior_z)
        logvar = self.prior_logvar(prior_z)

        return mean, logvar

    def decode(self, h_0, z_psi):
        x = torch.cat((h_0, z_psi), dim=1)
        # print("Decoder Cat", x.shape) # [2, 64, 32, 32]
        dec_x = self.decoder(x)

        return dec_x

    def gru_update(self, x_psi, z_psi, h_0):
        # print("GRU Input", x_psi.shape, z_psi.shape, h_0.shape) # [2, 32, 8, 8]
        concat_input = torch.cat((x_psi, z_psi, h_0), dim=1)
        # print("Concat Input", concat_input.shape) # [2, 48, 8, 8]

        processed_input = concat_input.reshape(-1, 48 * 8 * 8)
        # print("Processed Input", processed_input.shape)  # [2, 6144]

        h_next = self.rnn(processed_input)
        # print("h next", h_next.shape) # [2, 96, 32, 32]

        processed_h_next = h_next.reshape(-1, 16, 8, 8)
        # print("processed h next", processed_h_next.shape)  # [2, 16, 8, 8]

        return processed_h_next

    def loss_fn(self, x, x_reconstructed, mean, logvar):
        beta = 0.01

        # Reconstruction Loss
        loss_fn = nn.MSELoss()
        recon_loss = loss_fn(x_reconstructed, x)
        # print("recon", recon_loss)

        # KLD loss between prior and encoder
        kld = kld = -0.5 * torch.sum(logvar + 1 - mean.pow(2) - logvar.exp())
        # print("kld", kld)

        # Total loss
        loss = beta * kld + recon_loss
        # print("Loss:", recon_loss, kld, beta*kld)

        return loss

    def recon_loss_fn(self, x, x_reconstructed):
        # Reconstruction Loss
        loss_fn = nn.MSELoss()
        recon_loss = loss_fn(x_reconstructed, x)
        # print("recon", recon_loss)

        return recon_loss

    def kld_loss_fn(self, mean, logvar):
        beta = 1

        # KLD loss between prior and encoder
        # print("kld loss in", mean.shape, logvar.shape)
        mean = mean.flatten(1)
        logvar = logvar.flatten(1)
        # print("kld flatten", mean.shape, logvar.shape)

        kld = -0.5 * torch.sum(logvar + 1 - mean.pow(2) - logvar.exp(), dim=1)
        # print("kld per sample", kld.shape, kld)

        kld = torch.mean(kld, dim=0)
        # print("kld per sample out", kld.shape, kld)

        # Total loss
        # loss = beta * kld + recon_loss
        # print("Loss:", recon_loss, kld, beta*kld)

        return kld

    def double_kld_loss_fn(self, enc_mean, enc_logvar, prior_mean, prior_logvar):
        beta = 1

        # KLD loss between prior and encoder
        # print("kld loss in", mean.shape, logvar.shape)
        enc_mean = enc_mean.flatten(1)
        enc_logvar = enc_logvar.flatten(1)

        prior_mean = prior_mean.flatten(1)
        prior_logvar = prior_logvar.flatten(1)
        # print("kld flatten", enc_mean.shape, enc_logvar.shape, prior_mean.shape, prior_logvar.shape)

        # print("kld per sample", kld.shape, kld)
        kld_component = (2 * (enc_logvar - prior_logvar)) + \
                        (prior_logvar.exp().pow(2) + (prior_mean - enc_mean).pow(2)) / enc_logvar.exp().pow(2) - 1

        kld_per_sample = 0.5 * torch.sum(kld_component, dim=1)
        # print("kld per sample", kld_per_sample.shape)

        kld = torch.mean(kld_per_sample, dim=0)
        # print("kld per sample out", kld.shape, kld)

        # Total loss
        # loss = beta * kld + recon_loss
        # print("Loss:", recon_loss, kld, beta*kld)

        return kld

    def kld_normal(self, kld, prediction):

        # print("KLD Normal", prediction.shape)
        normalizer = prediction.flatten(1)
        # print("KLD Normal flatten", normalizer.shape)
        size = normalizer.size(1)
        # print("KLD Normal out", size)

        kld_out = kld/size
        # print("Final KLD", kld_out)

        return kld_out

    def forward(self, x, steps=27):
        # print("BAtch:", torch.Size(list(x.shape))) # [2, 32, 3, 32, 32]
        # print("Len Seq:", x.size(1))  # 32
        n_steps = x.size(1) - 1

        batch_loss = 0
        forecast_loss = []
        output = []
        comb_output = torch.empty((2, n_steps, 3, 32, 32)).to(device)
        batch_loss = 0

        # Pre-process input frames
        # x = self.concat_rgb(x)
        # print("Converted Input", x.shape) # [2, 15, 32, 32] 10*3

        h_0 = torch.zeros((2, 16, 8, 8), dtype=torch.float32).to(device)
        # print("h_0", h_0.shape) # [2, 16, 8, 8]

        for frame in range(n_steps):
            # print(frame)
            if frame < n_steps: # 6
                # print("Using actual frame")
                current_frame = x[:, frame, :]
            else:
                # print("Using predicted frame")
                current_frame = dec_x

            target_frame = x[:, frame+1, :]
            # print("Current Frame", current_frame.shape) # [2, 3, 32, 32]

            processed_frame = self.psi_x(current_frame)
            # print("Processed Frame", processed_frame.shape)  # [2, 16, 8, 8]

            z_mean, z_logvar = self.encode(processed_frame, h_0)
            # print("z_mean, z_logvar", z_mean.shape, z_logvar.shape) # [2, 64, 8, 8]

            prior_mean, prior_logvar = self.get_prior(h_0)
            # print("prior_mean, prior_logvar", prior_mean.shape, prior_logvar.shape)  # [2, 32, 8, 8]

            z = self.reparameterisation(z_mean, z_logvar)
            # print("Encoder out", z.shape) # [2, 64, 8, 8]

            z_psi = self.psi_z(z)
            # print("Feature Extracted z", z_psi.shape)  # [2, 32, 8, 8]

            dec_x = self.decode(h_0, z_psi)
            # print("Decoder Out:", dec_x.shape)  # [2, 3, 32, 32]
            output.append(dec_x)
            comb_output[:, frame] = dec_x

            # Frame loss
            frame_loss = self.recon_loss_fn(target_frame, dec_x)
            # print("Frame Loss", frame_loss.item())
            batch_loss += frame_loss

            forecast_loss.append(frame_loss.item())

            h_0 = self.gru_update(processed_frame, z_psi, h_0)

            # loss_int = self.loss_fn(target[frame].unsqueeze(0), dec_x, mean, logvar)
            # forecast_loss.append(loss_int.item())
            # batch_loss += loss_int
            # print("Loss:", batch_loss.item())  # [12, 3, 32, 32]

        # kld_per_sample = self.kld_loss_fn(z_mean, z_logvar)
        double_kld_per_sample = self.double_kld_loss_fn(z_mean, z_logvar, prior_mean, prior_logvar)
        # print("Compare KLD", kld_per_sample, double_kld_per_sample)
        kld_out = self.kld_normal(kld=double_kld_per_sample, prediction=comb_output)

        # print("Combined Out", comb_output.shape)
        # comb_loss = self.recon_loss_fn(x=comb_output, x_reconstructed=target)
        full_loss = kld_out + batch_loss
        # print("Full loss:", full_loss.item(), kld_out.item(), batch_loss.item())

        # output = torch.cat(output, dim=0)
        # print("Output:", output.shape)  # [12, 3, 32, 32]

        output = comb_output[0]

        return output, full_loss, forecast_loss

    def sample(self, n_steps):
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

def train_hnn():
    """
    Function to train the hnn model.
    """

    # Create dataset
    trainloader, testloader = get_data()

    torch.cuda.empty_cache()
    print("Loading model")

    # Initialise model

    model = ModelB(input_dim=3).to(device)
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

    for epoch in range(15):
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
            input_sequence = x_in # [:, :input_length, :]  # [2, 32, 3, 32, 32]
            target_sequence = x_in[:, 1:, :]  # [2, 31, 3, 32, 32]
            # print("Input Split:", total_length, input_sequence.shape, target_sequence.shape)

            # Pass through Model
            optimizer.zero_grad()

            output, batch_loss, forecast_loss = model(input_sequence)
            # loss = loss_fn(x_out, input_sequence)

            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()

            # if id % 249 == 0 and epoch == 9:
            #     sample_output, sample_comb = model.sample(n_steps=20)
            #     sample_comb = sample_comb.permute(0, 2, 3, 1).cpu().detach().numpy()
            #     visualize_rollout(sample_comb)



        ##############
        # Test

        model.eval()
        for id, test_batch in enumerate(testloader):
            x_in_test = torch.squeeze(test_batch, dim=0).to(device)

            # Split into input sequence and target
            total_length_test = x_in_test.shape[0]  # 32
            input_length_test = 5
            input_sequence_test = x_in_test# [:, :input_length_test, :]  # [20, 1, 32, 32]
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

    visualize_rollout(target_sequence_test)
    # visualize_test(forecast_loss_test)
    visualize_rollout(output_test)

    print(forecast_loss)
    print(forecast_loss_test)

    visualize_test(forecast_loss)
    visualize_test(forecast_loss_test)

    get_frames(sequence=target_sequence)
    get_frames(sequence=output)
    get_frames(sequence=target_sequence_test, n_steps=45)
    get_frames(sequence=output_test, n_steps=45)

    return

def visualize_test(test_losses):
    with torch.no_grad():
        x_vals = [i for i in range(len(test_losses))]
        plt.plot(x_vals, test_losses)
        plt.xlabel("Forecast Step")
        plt.ylabel("MSE Loss")
        plt.xlim(0, len(x_vals))
        plt.ylim(bottom=0)
        plt.show()

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

    train_hnn()