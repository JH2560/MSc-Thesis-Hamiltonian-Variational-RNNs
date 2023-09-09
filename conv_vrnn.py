import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from hnn import HNN_mlp
from Library.hgn_source.environments.environment import Environment, visualize_rollout
from Library.hgn_source.environments.datasets import EnvironmentSampler, EnvironmentLoader
from Library.hgn_source.environments.pendulum import Pendulum
from Library.hgn_source.environments.spring import Spring

def get_data():
    """
    Obtains the raw data and splits into train and test splits.
    """

    pd = Pendulum(mass=0.5, length=1, g=3)

    trainDS = EnvironmentSampler(environment=pd,
                                 dataset_len=50,
                                 number_of_frames=64,
                                 delta_time=.1,
                                 number_of_rollouts=1,
                                 img_size=32,
                                 color=False,
                                 noise_level=0.,
                                 radius_bound=(1.3, 2.3),
                                 seed=None)

    # Dataloader instance test, batch_mode disabled
    train = torch.utils.data.DataLoader(trainDS,
                                        shuffle=False,
                                        batch_size=None)

    testDS = EnvironmentSampler(environment=pd,
                                 dataset_len=1,
                                 number_of_frames=100,
                                 delta_time=.1,
                                 number_of_rollouts=1,
                                 img_size=32,
                                 color=False,
                                 noise_level=0.,
                                 radius_bound=(1.3, 2.3),
                                 seed=36)

    # Dataloader instance test, batch_mode disabled
    test = torch.utils.data.DataLoader(testDS,
                                        shuffle=False,
                                        batch_size=None)

    return train, test

def input_data(sequence, window):
    output = []
    L = len(sequence)
    # print(L)

    for i in range(L - window):
        win = sequence[i:i+window] # Grab values in window from i to i+window
        label = sequence[i+window: i+window+1] # Grab last item of window + 1 -> Item we want to predict
        # print(i)

        output.append((win, label))  # Create batches of sequences of size = window

    return output

class ConvVRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=128):
        super().__init__()
        self.input_dim = input_dim   # 40
        self.hidden_dim = hidden_dim # 128
        self.latent_dim = latent_dim #128
        # self.num_layers = num_layers

        # Feature extracting networks for x and z
        self.psi_x = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
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
            nn.Linear(self.hidden_dim * 32 * 32 + self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.enc_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.enc_log_var = nn.Linear(self.hidden_dim, self.latent_dim)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.fc2 = nn.Linear(in_features=self.hidden_dim, out_features=128 * 32 * 32 - self.hidden_dim)

        # self.fc_alter = nn.Linear(self.hidden_dim * 32 * 32 + self.hidden_dim, self.hidden_dim * 32 * 32)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid(),
            # nn.Linear(self.hidden_dim * 32 * 32 + self.hidden_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            # nn.LeakyReLU(0.2),
            # nn.Linear(self.hidden_dim, self.hidden_dim * 32 * 32),
            # nn.BatchNorm1d(self.hidden_dim * 32 * 32),
            # nn.LeakyReLU(0.2),
        )
        # self.dec_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        # self.dec_log_var = nn.Linear(self.hidden_dim, self.input_dim)

        # Recurrent Unit
        self.gru = nn.GRUCell(128 * 32 * 32 * 2 - self.hidden_dim, self.hidden_dim)

    def encode(self, x):
        # print(x.shape)
        z = self.encoder(x)
        # print("z", z.shape)

        # z = z.reshape(-1, 128 * 32 * 32)

        z_hidden = self.fc1(z)

        # Encode into mean and logvar
        mean = self.enc_mean(z_hidden)
        logvar = self.enc_log_var(z_hidden)

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
        # x_out = x_out.reshape(-1, 1, 32, 32)
        # x_out = x_out[:, 0]
        # print("xout", x_out.shape)

        return x_out

    def forward(self, x, hidden, future=0):
        outputs = []

        # x = x.split(1, dim=1)
        # print(x.shape)
        # x_in = x[:-1]    # [99, 1, 32, 32]
        # x_label = x[-1] # [1, 32, 32]
        # print(x_in.shape, x_label.shape)
        # x = x.view(-1, 1)
        # print(x.shape)
        # print(x)

        # Set Parameter for GRU Cell
        # print(x.shape[0])
        h_t = hidden #torch.zeros(x.shape[0], self.hidden_dim, dtype=torch.float32).to(device)

        # Obtain X
        # print("pre x", x.shape)
        x = self.psi_x(x)
        x = x.reshape(-1, 128 * 32 * 32)
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
        z_t = self.fc2(z_t)
        # z_t = z_t.reshape(-1, 128, 32, 32)

        # Sample x from decoder
        #h_t = h_t.permute(1, 0)
        # print("z_t", z_t.shape)
        # print("h", h_t.shape)
        f_t = torch.cat([h_t, z_t], dim=1)
        # print("f_t", f_t.shape)
        f_t = f_t.reshape(-1, 128, 32, 32)
        # print("f_t", f_t.shape)
        #x_out_alter = self.fc_alter(torch.cat([h_t, z_t], dim=1))
        #x_out_alter = x_out_alter.reshape(-1, 128, 32, 32)
        #x_out_alter = x_out_alter.reshape(-1, 128, 32, 32)
        # print(x_out_alter.shape)
        x_out = self.decode(f_t)
        prediction = x_out[-1]
        # print("xoutd", x_out.shape)
        # print("pred", prediction.shape)

        # Update Recurrent cell
        # print("HCell", x.shape, z_t.shape, torch.cat([x,z_t], dim=1).shape, h_t.shape)
        h_next = self.gru(torch.cat([x,z_t], dim=1), h_t)

        return prediction, h_next, z_prior, z_inferred, prior_mean, prior_logvar, enc_mean, enc_logvar


def loss_function(x, x_reconstructed, prior_mean, prior_logvar, enc_mean, enc_logvar):
    beta = 0.01

    # Loss between x_t and reconstructed x_t
    #print(x.shape, x_reconstructed.shape)
    # print(x)
    # x = x.type(torch.LongTensor)
    #print(x, x_reconstructed)
    loss_fn = nn.BCELoss()
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

def conv_vrnn_train(future=0):

    # Create dataset
    trainloader, testloader = get_data()

    torch.cuda.empty_cache()
    print("Loading model")

    model = ConvVRNN(1, 128, 128).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    outputs = dict()
    gen_outputs = dict()
    # hidden_states = []
    h_0 = torch.zeros(64, model.hidden_dim, dtype=torch.float32).to(device) # Batch size x Hidden dim
    # hidden_states.append(h_0)
    # print(hidden_states)

    # Training Loop
    print("Commencing training loop")
    training_loss = []
    generate_loss = []
    for epoch in range(4):
        # print(epoch)
        running_loss = 0

        # for batch in enumerate(train_input):
        #     print(batch)

        # Train with batches
        for id, batch in enumerate(trainloader):
            # print(seq)
            x = torch.squeeze(batch, dim=0).to(device) # [100, 1, 32, 32]
            # print(x.shape)

            input_dic = input_data(x, 20)
            # print(len(input_dic)) # Len = 64 - 20 = 44

            for seq, y_train in input_dic:
                #print("seq", seq.shape)
                seq = seq.to(device)
                y_train = torch.squeeze(y_train, dim=0).to(device)

                optimizer.zero_grad()
                pred, h_next, z_prior, z_inferred, prior_mean, prior_logvar, enc_mean, enc_logvar = model(seq, h_0)
                #print("predshape", pred.shape)

                # Update hidden
                h_0 = h_next.detach()
                # h_0 = h_t
                # print(h_0)

                loss = loss_function(y_train, pred, prior_mean, prior_logvar, enc_mean, enc_logvar)
                # print("loss", loss)

                running_loss += loss

                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), 1e-3)
                optimizer.step()

        outputs[epoch + 1] = {'x_in': x, 'out': pred}

        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", running_loss / 100)
        training_loss.append(running_loss.cpu().detach().numpy())

        # Predict Future
        for id, pred_data in enumerate(testloader):
            # print(id, pred_data.shape)
            pred_data = torch.squeeze(pred_data, dim=0).to(device)  # [100, 1, 32, 32]
            pred_in = pred_data[:-future]  # [100-future, 1, 32, 32]
            # print("pred_in", pred_in.shape)
            # x_label = x[-1]  # [1, 32, 32]
            for f in range(future):
                input_seq = pred_in[-future:]
                # print("in_seq", input_seq.shape)
                with torch.no_grad():
                    h_0_fut = torch.zeros(20, model.hidden_dim, dtype=torch.float32).to(device)  # Batch size x Hidden dim

                    pred, h_next, z_prior, z_inferred, prior_mean, prior_logvar, enc_mean, enc_logvar = model(input_seq, h_0_fut)

                    # Update hidden
                    # h_0_fut = h_next.detach()

                    # Append predicted image to pred_in
                    pred = pred[None, :] # [1, 1, 32, 32]
                    # print("pred", pred.shape)
                    pred_in = torch.cat((pred_in, pred), dim=0)
                    # print("pred_out", pred_in.shape)

        gen_outputs[epoch + 1] = {'pred_in': pred_data[-future:], 'out': pred_in[-40:]}

        # print("final_pred", pred_in.shape) # [100, 1, 32, 32]
        gen_loss = loss_function(pred_data[-future:], pred_in[-future:], prior_mean, prior_logvar, enc_mean, enc_logvar)
        print("\tGen Loss: ", gen_loss)
        generate_loss.append(gen_loss.cpu().detach().numpy())


    # Plotting the training loss
    plt.plot(range(1, 4 + 1), training_loss)
    plt.xlabel("Number of epochs")
    plt.ylabel("Training Loss")
    plt.show()

    # Plotting the gen loss
    plt.plot(range(1, 4 + 1), generate_loss)
    plt.xlabel("Number of epochs")
    plt.ylabel("Gen Loss")
    plt.show()

    return outputs, gen_outputs # x, x_out

def show_image(x, idx=0):
    x = x.view(100, 32, 32)

    fig = plt.figure()
    plt.imshow(x[idx].detach().numpy())
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

    outputs, gen_outputs = conv_vrnn_train(future=20)
    # imgs = outputs[10]["out"].cpu()
    # # print("outputs", imgs.shape)  # [100, 1, 32, 32]
    # # imgs = imgs.permute(0, 2, 3, 1)
    # # print("permuted outputs", imgs.shape)  # [100, 32, 32, 1]
    # imgs = imgs.detach().numpy()
    # visualize_rollout(imgs)
    #
    # real = outputs[10]["x_in"].cpu()
    # real = real.permute(0, 2, 3, 1).detach().numpy()
    # visualize_rollout(real)

    gen_imgs = gen_outputs[10]["out"].cpu()
    # print("outputs", gen_imgs.shape)  # [100, 1, 32, 32]
    gen_imgs = gen_imgs.permute(0, 2, 3, 1)
    # print("permuted outputs", imgs.shape)  # [100, 32, 32, 1]
    gen_imgs = gen_imgs.detach().numpy()
    visualize_rollout(gen_imgs)

    gen_real = gen_outputs[10]["pred_in"].cpu()
    gen_real = gen_real.permute(0, 2, 3, 1).detach().numpy()
    visualize_rollout(gen_real)

    # x, x_hat = conv_vrnn_train()
    #
    # show_image(x, idx=99)
    # show_image(x_hat, idx=99)



    # show_image(x, idx=99)
    # show_image(x_hat, idx=99)