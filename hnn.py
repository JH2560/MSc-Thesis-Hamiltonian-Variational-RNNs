from Library.hgn_source.environments.environment import Environment, visualize_rollout
from Library.hgn_source.environments.pendulum import Pendulum

import numpy as np
import torch
import torch.nn as nn

class hnn_mlp(nn.Module):
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

    def forward(self, x):
        """
        Forward pass of the MLP.
        """

        return self.layers(x.float())

if __name__ == "__main__":

    input_data = torch.tensor(np.random.rand(2))

    hnn = hnn_mlp(2)
    outputs = hnn(input_data)
    print(input_data, outputs)





    # pd = Pendulum(mass=.5, length=1, g=3)
    # rolls = pd.sample_random_rollouts(number_of_frames=100,
    #                                   delta_time=0.1,
    #                                   number_of_rollouts=16,
    #                                   img_size=32,
    #                                   noise_level=0.,
    #                                   radius_bound=(1.3, 2.3),
    #                                   color=True,
    #                                   seed=23)
    # idx = np.random.randint(rolls.shape[0])
    # visualize_rollout(rolls[idx])
