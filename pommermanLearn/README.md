# PommermanLearn

This repository contains all scripts used for training and testing agents in the Pommerman environment. It also contains many trained models + evaluation.

## Setup

Before you can run any of the training scripts, you have to install Python (3.6) and all required libraries first. Just navigate into the cloned github repository with:
 `cd playground` and execute `pip install -e .[extras]` to install Pommerman and all additional dependencies.

## How to train an agent:

Before you run the actual training script, it is possible to adjust hyper parameters by changing the value of the variables listed in the `params.py`file. At the end of the Readme there is an overview of the parameters you can adjust and their meaning.

To start the training of an agent you just have to run the `train_dqn.py` with:
`python train_dqn.py`

After this the training will run for the specified numbers of iterations and print intermediate results of the current run.

## Tensorboard logging

You can see the results of the training (also during the training) by running:
`tensorboard --logdir <directory>`

Just specify the correct directory where the logs are saved, the default is `playground/pommermanLearn/data/tensorboard`, depending where you run your script from you have to adjust the path accordingly.

Then you can open `http://localhost:6006/` in a browser of your choice to see your runs

Currently the following values are logged during training: 

- Win and Tie Ratio during training and evaluation
- Average Steps taken during training and evaluation
- Average Reward during training and evaluation
- Average Loss during training

## How to evaluate a model

If you want to evaluate a trained model you can just run the `test_model_seperate.py` file with `python test_model_seperate.py`, located in the `pommermanLearn` directory. Before you run it, you have to specify 2 paths for each trained agent by editing the corresponding variables `model_dir1` and `model_dir2` in `test_model_seprate.py`. Furthermore the configuration in `params.py` must match the configuration used during training. After running the script you will see the ratio for wins, ties and losses for the specified agents.

## Table with all hyperparameters and their meaning

| Variable                | Meaning                                                      |
| ----------------------- | ------------------------------------------------------------ |
| num_iterations          | Number of training iterations                                |
| episodes_per_iter       | Number of episodes per iteration                             |
| gradient_steps_per_iter | Number of gradient steps per iteration                       |
| batch_size              | Batch size for training                                      |
| episodes_per_eval       | Number of episodes to run for evaluation                     |
| intermediate_test       | After how many iterations an evaluation is done              |
| centralize_planes       | Whether the planes are centralized on the agent              |
| render_tests            | Whether to render the tests during evaluation                |
| env                     | Environment name to use for training                         |
| crop_fog                | Whether to crop the fog out of input                         |
| double_q                | Whether to use double den                                    |
| prioritized_replay      | Whether to use prioritized replay                            |
| beta                    | Determines how replays should be weighted in prioritized replay |
| device                  | Device to use with pytorch                                   |
| run_name                | Name of the folder where logs are saved                      |
| alpha                   | Determines how strongly prioritized replay is used           |
| categorical             | Whether to use categorical DQN                               |
| atom_size               | Number of atoms building the support vector used  in Categorical DQN                                                            |
| v_min                   | Minimum expected reward (for building the support vector)                                                        |
| v_max                   | Maximum expected reward (for building the support vector)                                  |
| dueling                 | Whether to use dueling DQN                                   |
| noisy                   | Whether to use noisy layers                                  |
| use_nstep               | Whether to use N Step learning                               |
| nsteps                  | Number of steps to use for N Step learning                   |
| communicate             | Communication mode (0, 1, 2) 0 = no communication            |
| use_memory              | Whether to use memory for the board                          |
| seed                    | Random seed to use                                           |
| gamma                   | Discount factor for rewards                                  |
| tau                     | Soft target network update rate                                                             |
| lr_q                    | Learning rate for Q-value                                    |
| exploration_noise       | Exploration noise for epsilon greedy                         |
| set_position            | Whether initial agent position is set on the board           |
| replay_size             | Size of replay buffer (must be a power of 2 to be compatible with prioritized replay) |
| max_steps               | Maximum number of steps for a game                           |
| reward_func             | Which reward function to use                                 |
| fifo_size               | Size of FIFO for step reward                                 |
| kill_rwd                | Reward for killing an enemy                                  |
| teamkill_rwd            | Reward for killing a teammate                                |
| death_rwd               | Reward for dying                                             |
| win_loss_bonus          | Reward for winning                                           |
| step_rwd                | Reward for taking steps not in FIFO                          |
| item_rwd                | Reward for picking up an item                                |
| bomb_tracker            | Whether to use bomb tracker                                  |
| memory_method           | Memory method to use                                         |
| forgetfullness          | Percentage decrease of confidence in the memory                                                             |
| normalize_steps         | Whether to normalize steps                                   |

## Code References

https://github.com/Curt-Park/rainbow-is-all-you-need (Used for Categorical DQN, Noisy Layers, Multi-Step Learning) (dqn.py, models.py, replay_buffer.py))

https://github.com/KaleabTessera/DQN-Atari (Some parts of our DQN implementation are based on this)

https://github.com/BorealisAI/pommerman-baseline (Action filter and some agent are taken from this) (action_prune.py, skynet_agents.py)

https://nn.labml.ai/rl/dqn/replay_buffer.html (Implementation of Prioritized Replay Buffer is based on this)
