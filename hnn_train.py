import numpy as np
import torch
import torch.nn as nn

from hnn import HNN_mlp
from Library.hgn_source.environments.environment import Environment, visualize_rollout
from Library.hgn_source.environments.pendulum import Pendulum

def get_data():
    """
    Obtains the raw data and splits into train and test splits.
    """

    # Initialise pendulum for samples
    pd = Pendulum(mass=.5, length=1, g=3)

    # Generate N rollouts/trajectories and add to dataset
    data = []

    for rollout in range(5):
        q, p = pd.sample_random_rollouts(number_of_frames=64,
                                      delta_time=0.1,
                                      number_of_rollouts=16,
                                      img_size=32,
                                      noise_level=0.,
                                      radius_bound=(1.3, 2.3),
                                      color=True,
                                      )
        #print(len(q))

        data.append(np.stack((q, p)).T)

    # Concatenate
    data = np.concatenate(data)

    # Make train/test split
    split_value = int(len(data) * 0.7)

    train_data = []
    test_data = []

    train_data, test_data = data[:split_value], data[split_value:]

    # Convert NumPy arrays to tensors
    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data)

    # Set Requires.grad parameter
    train_data.requires_grad = True
    test_data.requires_grad = True

    return train_data, test_data

def train_hnn():
    """
    Function to train the hnn model.
    """

    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Initialise model
    hnn_model = HNN_mlp(input_dim=2)

    # Define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(hnn_model.parameters(), lr=1e-3)

    # Get data
    train_data, test_data = get_data()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    # Training loop
    training_loss = []
    #test_loss = []

    for epoch in range(5):
        print("Commencing epoch {}".format(epoch))

        # Initialise loss
        Curr_loss = 0.0

        # Zero Gradients from previous epochs
        optimizer.zero_grad()

        # Iterate over DataLoader for training
        for id, batch in enumerate(train_loader):
            # Position and Momentum Data
            pos_data = batch[:, 0]
            mom_data = batch[:, 1]

            # Pass batch through HNN
            output = hnn_model(batch)

            # Obtain gradients and next predicted q and p
            grad_outputs = hnn_model.get_gradient(batch, hnn_model)

            print(grad_outputs)








if __name__ == "__main__":
    # train_data, test_data = get_data()
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=False)
    # print(train_data)
    # print("-----")
    # print(train_loader)
    # print("-----")
    # train_features = next(iter(train_loader))
    # print(train_features)
    # print("-----")
    # for i, data in enumerate(train_loader):
    #     pos_data = data[:,0]
    #     mom_data = data[:,1]
    #     print(i, data)
    #     print(pos_data, mom_data)
    train_hnn()