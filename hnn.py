from Library.hgn_source.environments.environment import Environment, visualize_rollout
from Library.hgn_source.environments.pendulum import Pendulum

import numpy as np
import torch
import torch.nn as nn

class HNN_mlp(nn.Module):
    """
    Basic implementation of a HNN as an MLP. To handle direct position and momentum datapoints as input.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 32),
            nn.Softplus(),
            nn.Linear(32, 1),
        )

        # self.layers = nn.Sequential(
        #     nn.Linear(input_dim, 8),
        #     # nn.Softplus(),
        #     nn.Linear(8, 8),
        #     # nn.Softplus(),
        #     nn.Linear(8, 4),
        #     # nn.Softplus(),
        #     nn.Linear(4, 1),
        #     nn.Softplus(),
        # )

    def get_gradient(self, q, p, hnn):
        """
        Obtain position and momentum gradients for use in Euler step.
        """

        # Combine inputs
        # x = torch.concat((q, p))

        # Obtain position, momentum, and energy
        energy = hnn(q, p)
        #print(energy)

        # Obtain Gradients of Energy
        dh = torch.autograd.grad(energy, (q, p), create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(energy))
        #print("DH: {}".format(dh))

        # Obtain Momentum gradient: dp/dt = -dH/dq
        dp = -dh[1]

        # Obtain Position gradient: dq/dt = dH/dp
        dq = dh[0]

        # dp = torch.autograd.grad(energy, x, grad_outputs=torch.ones_like(energy), allow_unused=True)
        #dq = torch.autograd.grad(energy, p, grad_outputs=torch.ones_like(energy), allow_unused=True)

        return dq, dp

    def euler_step(self, q, p, dq, dp):
        """
        Computes successor position and momentum.
        """

        # Combine inputs
        # x = torch.concat((q.reshape(1), p.reshape(1)))

        # Define delta t:
        delta_t = 0.1

        # Define position and momentum
        # q = x[0]
        # p = x[1]

        # Get gradients
        # dp, dq = self.get_gradient(x, hnn)

        # Euler step
        p_successor = p + delta_t * dp
        q_successor = q + delta_t * dq

        return q_successor, p_successor

    def forward(self, q, p):
        """
        Forward pass of the MLP.
        """

        # Combine inputs
        x = torch.concat((q.reshape(1), p.reshape(1)), dim=0)

        # Obtain energy
        energy = self.layers(x.float())

        # Perform Euler step
        # q_successor, p_successor = self.euler_step(x)

        return energy

if __name__ == "__main__":

    input_data = torch.tensor(np.random.rand(2), requires_grad=True)
    pos, mom = input_data[0].reshape(1), input_data[1].reshape(1)
    print(pos, mom)

    hnn = HNN_mlp(2)
    outputs = hnn(pos, mom)
    print(input_data, outputs)
    grad_outputs = hnn.get_gradient(pos, mom, hnn)
    print(input_data, grad_outputs)





    # pd = Pendulum(mass=.5, length=1, g=3)
    # rolls = pd.sample_random_rollouts(number_of_frames=100,
    #                                   delta_time=0.1,
    #                                   number_of_rollouts=16,
    #                                   img_size=32,
    #                                   noise_level=0.,
    #                                   radius_bound=(1.3, 2.3),
    #                                   color=True,
    #                                   seed=23)
    # # idx = np.random.randint(rolls.shape[0])
    # # visualize_rollout(rolls[idx])
    # print(rolls.shape)
