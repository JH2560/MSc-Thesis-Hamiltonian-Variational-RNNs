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
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def get_gradient(self, x, hnn):
        """
        Obtain position and momentum gradients for use in Euler step.
        """

        # Obtain position, momentum, and energy
        energy = hnn(x)
        #print(energy)

        # Obtain Gradients of Energy
        dh = torch.autograd.grad(energy, x, grad_outputs=torch.ones_like(energy))
        #print(dh)

        # Obtain Position gradient: dp/dt = -dH/dq
        dp = -dh[0][1]

        # Obtain Momentum gradient: dq/dt = dH/dp
        dq = dh[0][0]

        # dp = torch.autograd.grad(energy, x, grad_outputs=torch.ones_like(energy), allow_unused=True)
        #dq = torch.autograd.grad(energy, p, grad_outputs=torch.ones_like(energy), allow_unused=True)

        return dp, dq

    def euler_step(self, x, hnn):
        """
        Computes successor position and momentum.
        """

        # Define delta t:
        delta_t = 0.1

        # Define position and momentum
        q = x[0]
        p = x[1]

        # Get gradients
        dp, dq = self.get_gradient(x, hnn)

        # Euler step
        p_successor = p + delta_t * dp
        q_successor = q + delta_t * dq

        return q_successor, p_successor

    def forward(self, q, p):
        """
        Forward pass of the MLP.
        """

        # Obtain energy
        energy = self.layers(x.float())

        # Perform Euler step
        # q_successor, p_successor = self.euler_step(x)


        return self.layers(x.float())

if __name__ == "__main__":

    input_data = torch.tensor(np.random.rand(2), requires_grad=True)

    hnn = HNN_mlp(2)
    outputs = hnn(input_data)
    print(input_data, outputs)
    grad_outputs = hnn.get_gradient(input_data, hnn)
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
