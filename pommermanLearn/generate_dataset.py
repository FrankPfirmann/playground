import logging
import numpy as np
import pommerman as pm
from pommerman.agents import SimpleAgent

from util.data import transform_observation

# TODO: Integrate back into train_embeddings.py

def setup_logger(log_level=logging.INFO):
    """
    Setup the global logger

    :param log_level: The minimum log level to display. Can be one of
        pythons built-in levels of the logging module.
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def play_game(env):
    """
    Play a game in the given environment and return the collected
    observations
    """
    observations=[]
    observation=env.reset()
    done=False

    while not done:
        act=env.act(observation)
        observation, rwd, done, _ = env.step(act)
        observations.append(observation)
        #env.render()
    env.close()

    return observations

def generate_dataset(num_games, flatten=True):
    X=[]
    for i in range(num_games):
        # Generate data by playing games with simple agents
        agent_list=[
            SimpleAgent(),
            SimpleAgent(),
            SimpleAgent(),
            SimpleAgent()
        ]

        env = pm.make("PommeRadioCompetition-v2", agent_list)
        observations=play_game(env)
        for observation in observations:
            for view in observation:
                planes = transform_observation(view, p_obs=True, centralized=True)
                if flatten:
                    x = planes.flatten()
                else:
                    x = planes
                X.append(np.array(x, dtype=np.float32))
        logging.info(f"Finished game {i+1}/{num_games} in {len(observations)} steps")
    return X

setup_logger()

dataset=generate_dataset(100, flatten=False)
np.save('./data/4simple-100games-planes.pkl', dataset, allow_pickle=True)