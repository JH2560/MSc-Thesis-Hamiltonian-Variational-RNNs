#Library/HGN Source/environments/pendulum.py
from Library.hgn_source.environments.pendulum import Pendulum
from Library.hgn_source.environments.environment import visualize_rollout
import numpy as np


if __name__ == "__main__":

    pd = Pendulum(mass=.5, length=1, g=3)
    rolls = pd.sample_random_rollouts(number_of_frames=100,
                                      delta_time=0.1,
                                      number_of_rollouts=16,
                                      img_size=32,
                                      noise_level=0.,
                                      radius_bound=(1.3, 2.3),
                                      color=True,
                                      seed=23)
    idx = np.random.randint(rolls.shape[0])
    print(rolls[idx])
    visualize_rollout(rolls[idx])
