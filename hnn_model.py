from Library.hgn_source.environments.environment import Environment, visualize_rollout
from Library.hgn_source.environments.pendulum import Pendulum

import numpy as np
import torch
import torch.nn as nn

class HNN(nn.Module):
    """
    Basic implementation of a HNN as a Conv Net. To handle direct position and momentum datapoints as input.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(0.1),
        )

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
        dp_1 = -dh[1]
        # dp_2 = -torch.autograd.grad(energy, q, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(energy))[0]

        # Obtain Position gradient: dq/dt = dH/dp
        dq_1 = dh[0]
        # dq_2 = torch.autograd.grad(energy, p, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(energy))[0]

        # print(dq_1[0][0][0])

        # dp = torch.autograd.grad(energy, x, grad_outputs=torch.ones_like(energy), allow_unused=True)
        #dq = torch.autograd.grad(energy, p, grad_outputs=torch.ones_like(energy), allow_unused=True)

        # print("Energy:", energy) # 1.4511
        # print("Euler", dq_1.shape, dq_1) # [1, 32, 32, 32]
        # print("Q", q.shape, q)

        return dq_1, dp_1

    def euler_step(self, q, p, hnn):
        """
        Computes successor position and momentum.
        """

        # Combine inputs
        # x = torch.concat((q.reshape(1), p.reshape(1)))

        # Define delta t:
        delta_t = 1

        # Define position and momentum
        # q = x[0]
        # p = x[1]

        # Get gradients
        # dp, dq = self.get_gradient(x, hnn)
        dq, dp = self.get_gradient(q, p, hnn)
        # print(dq)

        # Euler step
        p_successor = p + delta_t * dp
        q_successor = q + delta_t * dq
        # print("Euler:", q[0], dq[0], q_successor[0])
        # print("")

        return q_successor, p_successor

    def forward(self, q, p):
        """
        Forward pass of the HNN.
        """

        ################
        # Single Frame #
        ################

        # Combine inputs:
        x = torch.cat((q, p), dim=1)
        # print("X:", q.shape, x.shape)

        energy = self.layers(x.float())  # [1, 1, 1]
        energy = energy.squeeze(dim=1).squeeze(dim=1) # [1]


        return energy
