import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from hnn import HNN_mlp
from Library.hgn_source.environments.environment import Environment, visualize_rollout
from Library.hgn_source.environments.datasets import EnvironmentSampler, EnvironmentLoader
from Library.hgn_source.environments.pendulum import Pendulum
from Library.hgn_source.environments.spring import Spring

def create_dataset():
    x = torch.linspace(0, 799, 800) # Values from 0 to 799
    y = torch.sin(x * 2 * np.pi / 40)

    # plt.figure(figsize=(12, 4))
    # plt.xlim(-10, 801)
    # plt.grid(True)
    # plt.plot(y.numpy())

    test_size = 40
    train_set = y[:-test_size]  # Values from 0 - 759
    test_set = y[-test_size:] # Values from 760 - 799

    return train_set, test_set, y

def input_data(sequence, window):
    output = []
    L = len(sequence)

    for i in range(L - window):
        win = sequence[i:i+window] # Grab values in window from i to i+window
        label = sequence[i+window: i+window+1] # Grab last item of window + 1 -> Item we want to predict
        # print(i)

        output.append((win, label))  # Create batches of sequences of size = window

    return output

class VRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=128):
        super().__init__()
        self.input_dim = input_dim   # 40
        self.hidden_dim = hidden_dim # 128
        self.latent_dim = latent_dim #128
        # self.num_layers = num_layers

        # Feature extracting networks for x and z
        self.psi_x = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.psi_z = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Prior - Get Hyperparameters for KLD
        self.prior = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.prior_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.prior_log_var = nn.Linear(self.hidden_dim, self.latent_dim)

        # Encoder - Outputs mean and logvar. Keep hyperparameters for KLD.
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.enc_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.enc_log_var = nn.Linear(self.hidden_dim, self.latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, 40),
            nn.BatchNorm1d(40),
            nn.LeakyReLU(0.2),
        )
        # self.dec_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        # self.dec_log_var = nn.Linear(self.hidden_dim, self.input_dim)

        # Recurrent Unit
        self.gru = nn.GRUCell(self.hidden_dim * 2, self.hidden_dim)

    def encode(self, x):
        z = self.encoder(x)

        # Encode into mean and logvar
        mean = self.enc_mean(z)
        logvar = self.enc_log_var(z)

        return mean, logvar

    def prior_encode(self, x):
        z = self.prior(x)

        # Encode into mean and logvar
        mean = self.prior_mean(z)
        logvar = self.prior_log_var(z)

        return mean, logvar

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mean + logvar * eps

    def decode(self, z):

        x_out = self.decoder(z)
        x_out = x_out[:, 0]
        # print("xout", x_out.shape)

        return x_out

    def forward(self, x, future=0):
        outputs = []

        # x = x.split(1, dim=1)
        # print(x.shape)
        x = x.view(-1, 1)
        # print(x.shape)
        # print(x)

        # Set Parameter for GRU Cell
        # print(x.shape[0])
        h_t = torch.zeros(x.shape[0], self.hidden_dim, dtype=torch.float32)

        # Obtain X
        # print("pre x", x.shape)
        x = self.psi_x(x)
        # print("post x", x.shape)

        # Obtain Z from prior
        prior_mean, prior_logvar = self.prior_encode(h_t)
        z_prior = self.reparameterization(prior_mean, prior_logvar)

        # Obtain inferred Z from the encoder
        # print("x", x.shape)
        # print("h", h_t.shape)
        # h_t = h_t.permute(1, 0)
        # print(h_t.shape)
        # print(torch.cat([x, h_t], dim=1))
        enc_mean, enc_logvar = self.encode(torch.cat([x, h_t], dim=1))
        z_inferred = self.reparameterization(enc_mean, enc_logvar)

        # Sample from psi_z
        # print("pre z", z_inferred.shape)
        z_t = self.psi_z(z_inferred)
        # print("post z", z_inferred.shape)

        # Sample x from decoder
        #h_t = h_t.permute(1, 0)
        x_out = self.decode(torch.cat([h_t, z_t], dim=1))
        # print("xoutd", x_out.shape)

        # Update Recurrent cell
        h_next = self.gru(torch.cat([x,z_t], dim=1), h_t)

        return x_out, h_next, z_prior, z_inferred, prior_mean, prior_logvar, enc_mean, enc_logvar


def loss_function(x, x_reconstructed, prior_mean, prior_logvar, enc_mean, enc_logvar):
    beta = 0.01

    # Loss between x_t and reconstructed x_t
    #print(x.shape, x_reconstructed.shape)
    # print(x)
    # x = x.type(torch.LongTensor)
    #print(x, x_reconstructed)
    loss_fn = nn.MSELoss()
    recon_loss = loss_fn(x_reconstructed, x)
    # print("recon", recon_loss)

    # KLD loss between prior and encoder
    kld_component = (2 * (enc_logvar - prior_logvar)) + \
                    (prior_logvar.exp().pow(2) + (prior_mean - enc_mean).pow(2))/enc_logvar.exp().pow(2) - 1

    kld = 0.5 * torch.sum(kld_component)
    # print("kld", kld)

    # Total loss
    loss = beta * kld + recon_loss

    return loss

def vrnn_train():

    # Create dataset
    train_set, test_set, y = create_dataset()
    train_data = input_data(train_set, 40)  # Create batches of 40
    # print("train data", len(train_data))

    model = VRNN(1, 128, 128)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    outputs = dict()

    # Training Loop
    training_loss = []
    for epoch in range(10):
        running_loss = 0

        # for batch in enumerate(train_input):
        #     print(batch)

        # Train with batches
        for seq, y_train in train_data:
            # print("seq", seq)

            optimizer.zero_grad()
            x_out, h_next, z_prior, z_inferred, prior_mean, prior_logvar, enc_mean, enc_logvar = model(seq)
            # print(seq.shape)
            # print(x_out)

            loss = loss_function(seq, x_out, prior_mean, prior_logvar, enc_mean, enc_logvar)
            # print("loss", loss)

            running_loss += loss

            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1e-3)
            optimizer.step()

            # for name, param in model.named_parameters():
            #     try:
            #         print(name, param.grad.norm())
            #     except:
            #         print(name)

            # for p, n in zip(model.parameters(), model._all_weights[0]):
            #     if n[:6] == 'weight':
            #         print('===========\ngradient:{}\n----------\n{}'.format(n, p.grad))

        # optimizer.zero_grad()
        #
        # outputs, enc_mean, enc_logvar = model(train_input)
        #
        # loss = loss_function(train_target, outputs, enc_mean, enc_logvar)
        # print("loss", loss.item())

        #running_loss += loss

        outputs[epoch + 1] = {'x_in': seq, 'out': x_out}

        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", running_loss / 100)
        training_loss.append(running_loss.detach().numpy())

    # Plotting the training loss
    plt.plot(range(1, 10 + 1), training_loss)
    plt.xlabel("Number of epochs")
    plt.ylabel("Training Loss")
    plt.show()

    return outputs

if __name__ == "__main__":

    train_set, test_set, y = create_dataset()
    train_data = input_data(train_set, 40)

    outputs = vrnn_train()
    print(outputs.keys())

    # Plot
    x = outputs[10]["x_in"]
    x_out = outputs[10]["out"]

    plt.figure(figsize=(12, 4))
    plt.xlim(700, 801)
    plt.grid(True)
    plt.plot(x)
    plt.plot(range(760, 800), x_out.detach().numpy())
    plt.show()



    # show_image(x, idx=99)
    # show_image(x_hat, idx=99)