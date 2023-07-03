import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from hnn import HNN_mlp
from Library.hgn_source.environments.environment import Environment, visualize_rollout
from Library.hgn_source.environments.pendulum import Pendulum
from Library.hgn_source.environments.spring import Spring

def get_data():
    """
    Obtains the raw data and splits into train and test splits.
    """

    # Initialise pendulum for samples
    # pd = Pendulum(mass=.5, length=1, g=3)
    sp = Spring(mass=.5, elastic_cst=2, damping_ratio=0.)

    # Generate N rollouts/trajectories and add to dataset
    data = []

    for rollout in range(50):
        q, p = sp.sample_random_rollouts(number_of_frames=64,
                                      delta_time=0.1,
                                      number_of_rollouts=16,
                                      img_size=32,
                                      noise_level=0.,
                                      radius_bound=(1.3, 2.3),
                                      color=True,
                                      )
        print(len(q))
        #print(q)

        data.append(np.stack((q, p)).T)

    # Concatenate
    data = np.concatenate(data)

    # Make train/test split
    # split_value = int(len(data) * 0.7)

    train_data = []
    test_data = []

    # train_data, test_data = data[:split_value], data[split_value:]
    train_data, test_data = data[:64*49], data[64*49:]

    # Convert NumPy arrays to tensors
    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data)

    # Set Requires.grad parameter
    train_data.requires_grad = True
    test_data.requires_grad = True
    print(len(train_data))

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
    optimizer = torch.optim.Adam(hnn_model.parameters(), lr=2e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    # Get data
    train_data, test_data = get_data()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=False)

    # Training loop
    training_loss_list = []
    test_loss_list = []

    #print(list(hnn_model.parameters())[0].grad)

    for epoch in range(50):
        #print("Commencing epoch {}".format(epoch))

        # # Initialise loss
        # curr_loss = 0.0

        # Zero Gradients from previous epochs
        #optimizer.zero_grad()



        # Iterate over DataLoader for training
        for id, batch in enumerate(train_loader):
            # Initialise loss
            curr_loss = 0.0

            optimizer.zero_grad()

            #print(id)
            # Position and Momentum Data
            pos_data = batch[:, 0]
            mom_data = batch[:, 1]
            #print(batch)

            # Store state evolution
            hnn_pos = []
            hnn_mom = []

            # Loop through items
            for item in range(len(pos_data)-1):

                # Store initial state
                if item == 0:
                    # print(len(pos_data))
                    q = pos_data[item]
                    p = mom_data[item]
                    # print(q,p)

                    hnn_pos.append(q)
                    hnn_mom.append(p)

                q = hnn_pos[-1]
                p = hnn_mom[-1]

                # Pass batch through HNN
                output = hnn_model(q, p)
                #print(output.shape)

                # Obtain gradients and next predicted q and p
                dq, dp = hnn_model.get_gradient(q, p, hnn_model)

                # Euler step
                q_successor, p_successor = hnn_model.euler_step(hnn_pos[-1], hnn_mom[-1], dq, dp)
                hnn_pos.append(q_successor)
                hnn_mom.append(p_successor)
                #print(q_successor)



            # Compute loss
            # print(hnn_pos[-1], hnn_mom[-1])
            #print(hnn_pos.dtype, hnn_mom.dtype)

            # hnn_pos = torch.DoubleTensor(hnn_pos)[None, :]
            # hnn_mom = torch.DoubleTensor(hnn_mom)[None, :]
            #x = torch.concat((hnn_pos, hnn_mom), dim=1)
            #print(torch.transpose(hnn_pos, 1, 0))
            #
            # hnn_pos = torch.transpose(hnn_pos, 1, 0)
            # hnn_mom = torch.transpose(hnn_mom, 1, 0)

            # x = torch.concat((hnn_pos, hnn_mom), dim=1)
            #print(x.requires_grad)
            #print(batch, x)
            z = torch.stack(hnn_pos)
            y = torch.stack(hnn_mom)

            x = torch.stack([z, y]).T
            #print(x.requires_grad)
            #print(x)
            loss = loss_fn(batch, x)
            # training_loss_list.append(loss)
            #print(loss)
            # print(loss.grad_fn, x.grad_fn, hnn_pos[0].grad_fn)
            # curr_loss += loss

            # Update items
            loss.backward()
            optimizer.step()


            #print(list(hnn_model.parameters())[0].grad)

            #print(hnn_model.parameters())
            # for param in hnn_model.parameters():
            #     if param.requires_grad:
            #         print(param.data)
            # for p in hnn_model.parameters():
            #     if p.grad is None:
            #         print(p.grad.data)

            # for name, param in hnn_model.named_parameters():
            #     print(name, param.grad)


        # Test loop
        #scheduler.step()

        # Position and Momentum Data
        test_pos_data = test_data[:, 0]
        test_mom_data = test_data[:, 1]
        #print(test_data)

        # Store state evolution
        test_hnn_pos = []
        test_hnn_mom = []

        # Loop through items
        for test_item in range(len(test_pos_data) - 1):
            # Store initial state
            if test_item == 0:
                # print(len(pos_data))
                test_q = test_pos_data[test_item]
                test_p = test_mom_data[test_item]
                # print(q,p)

                test_hnn_pos.append(test_q)
                test_hnn_mom.append(test_p)

            test_q = test_hnn_pos[-1]
            test_p = test_hnn_mom[-1]

            # Pass batch through HNN
            test_output = hnn_model(test_q, test_p)
            # print(output.shape)

            # Obtain gradients and next predicted q and p
            test_dq, test_dp = hnn_model.get_gradient(test_q, test_p, hnn_model)

            # Euler step
            test_q_successor, test_p_successor = hnn_model.euler_step(test_hnn_pos[-1], test_hnn_mom[-1], test_dq, test_dp)
            test_hnn_pos.append(test_q_successor)
            test_hnn_mom.append(test_p_successor)

        # Compute Loss
        test_z = torch.stack(test_hnn_pos)
        test_y = torch.stack(test_hnn_mom)

        test_x = torch.stack([test_z, test_y]).T
        # print(x.requires_grad)
        #print(test_x)
        test_loss = loss_fn(test_data, test_x)
        test_loss_list.append(test_loss)

        if epoch % 10 == 0:
            print(epoch, "th iteration : ", test_loss)

    return test_loss_list

def visualize_test(test_losses):
    with torch.no_grad():
        x_vals = [i for i in range(len(test_losses))]
        plt.plot(x_vals, test_losses)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.xlim(0, len(x_vals))
        plt.ylim(bottom=0)
        plt.show()














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
    #     print(pos_data)
    test_losses = train_hnn()
    visualize_test(test_losses)