# PommermanLearn

This repo contains all scripts used for training and testing agents in the Pommerman environment. It also contains many trained models + evaluation.

## Requirements:

Before you can run any of the training scripts, you have to install all required libraries first. Just navigate into the cloned github repo with:
 `cd playground` and execute `pip install -e .` to install the needed libraries.

## How to train an agent:

Before you run the actual training script, it is possible to adjust hyper parameters by changing the value of the variables listed in the `params.py`file. At the end of the Readme there is an overview of the parameters you can adjust and their meaning.

To start the training of an agent you just have to run the `main.py` with:
`python main.py`

After this the training will run for the specified numbers of iterations and print intermediate results of the current run.

## Tensorboard logging

You can see the results of the training (also during the training) by running:
`tensorboard --logdir <directory>`

Just specify the correct directory where the logs are saved, the default is `playground/pommermanLearn/data/tensorboard`, depending where you run your script from you have to adjust the path accordingly.

Then you can go to `http://localhost:6006/` to see your runs

Currently the following values are logged during training: 

- Win and Tie Ratio during training and evaluation
- Average Steps taken during training and evaluation
- Average Reward during training and evaluation
- Average Loss during training

## How to evaluate a model

If you want to evaluate a trained model you can just run the `test_model_seperate.py` file with `python test_model_seperate.py`, located in the `pommermanLearn` directory. Before you run it, you have to specify 2 paths for each trained agent by editing the corresponding variables `model_dir1` and `model_dir2` in `test_model_seprate.py`. After running the script you will see the ratio for wins, ties and losses for the specified agents.



## Table with all hyperparameters and their meaning

| Variable                | Meaning                                         |
| ----------------------- | ----------------------------------------------- |
| num_iterations          | Number of training iterations                   |
| episodes_per_iter       | Number of episodes per iteration                |
| gradient_steps_per_iter | Number of gradient steps per iteration          |
| batch_size              | Batch size for training                         |
| episodes_per_eval       | Number of episodes to run for evaluation        |
| intermediate_test       | After how many iterations an evaluation is done |
| centralize_planes       | Whether the planes are centralized on the agent |
| render_tests            | Whether to render the tests during evaluation   |
| env                     | Environment name to use for training            |
| crop_fog                |                                                 |
| double_q                |                                                 |
| prioritized_replay      |                                                 |
| beta                    |                                                 |
| device                  |                                                 |
| run_name                |                                                 |
| alpha                   |                                                 |
| beta                    |                                                 |
| categorical             |                                                 |
| atom_size               |                                                 |
| v_min                   |                                                 |
| v_max                   |                                                 |
| dueling                 |                                                 |
| noisy                   |                                                 |
| use_nstep               |                                                 |
| nsteps                  |                                                 |
| communicate             |                                                 |
| use_memory              |                                                 |
| seed                    |                                                 |
| gamma                   |                                                 |
| tau                     |                                                 |
| lr_q                    |                                                 |
| exploration_noise       |                                                 |
| set_position            |                                                 |
| replay_size             |                                                 |
| max_steps               |                                                 |
| reward_func             |                                                 |
| fifo_size               |                                                 |
| kill_rwd                |                                                 |
| teamkill_rwd            |                                                 |
| death_rwd               |                                                 |
| win_loss_bonus          |                                                 |
| step_rwd                |                                                 |
| item_rwd                |                                                 |
| bomb_tracker            |                                                 |
| memory_method           |                                                 |
| forgetfullness          |                                                 |
| normalize_steps         |                                                 |

