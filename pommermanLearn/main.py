#!/usr/bin/env python3
# solve error #15
import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse
import logging
import random
from datetime import datetime
import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pommerman.constants import Action

import params as p
from data_augmentation import DataAugmentor_v1
from data_generator import DataGeneratorPommerman
from dqn import DQN
from models import Pommer_Q
from util.analytics import Stopwatch
from util.data import transform_observation_simple, transform_observation_partial, transform_observation_centralized

def test_pommerman_dqn():
    torch.manual_seed(p.seed)
    np.random.seed(p.seed)
    random.seed(p.seed)

    if p.p_observable:
        transform_func = transform_observation_partial
    elif p.centralize_planes:
        transform_func = transform_observation_centralized
    else:
        transform_func = transform_observation_simple

    # Initialize 2 DQNs
    q1 = Pommer_Q(p.p_observable, transform_func)
    q_target1 = Pommer_Q(p.p_observable, transform_func)
    algo1 = DQN(q1, q_target1)
    q2 = Pommer_Q(p.p_observable, transform_func)
    q_target2 = Pommer_Q(p.p_observable, transform_func)
    algo2 = DQN(q2, q_target2)

    data_generator = DataGeneratorPommerman(
	p.env,
	augmenter=[
            #DataAugmentor_v1()
        ])

    run_name=datetime.now().strftime("%Y%m%dT%H%M%S")
    log_dir=os.path.join("./data/tensorboard/", run_name)
    logging.info(f"Staring run {run_name}")
    writer = SummaryWriter(log_dir=log_dir)

    for i in range(p.num_iterations):
        logging.info(f"Iteration {i+1}/{p.num_iterations} started")
        iteration_stopwatch = Stopwatch(start=True)
        policy1 = algo1.get_policy()
        policy2 = algo2.get_policy()

        res, ties, avg_rwd, act_counts, avg_steps = data_generator.generate(p.episodes_per_iter, policy1, policy2, q1.get_transformer())
        # Normalize act_counts
        act_counts[0] = [act/sum(act_counts[0]) for act in act_counts[0]]
        act_counts[1] = [act/sum(act_counts[1]) for act in act_counts[1]]
        #The agents wins are stored at index 0 i the data_generator
        win_ratio = res[0] / (sum(res)+ties)
        total_loss=0

        # fit models on generated data
        gradient_step_stopwatch=Stopwatch(start=True)
        for _ in range(p.gradient_steps_per_iter):
            if p.episode_backward:
                batch = data_generator.get_episode_buffer()
            else:
                batch = data_generator.get_batch_buffer(p.batch_size, 0)
            loss = algo1.train(batch)
            total_loss += loss

        for _ in range(p.gradient_steps_per_iter):
            if p.episode_backward:
                batch = data_generator.get_episode_buffer()
            else:
                batch = data_generator.get_batch_buffer(p.batch_size, 1)
            loss=algo2.train(batch)
            total_loss+=loss
        avg_loss=total_loss/p.gradient_steps_per_iter
        logging.debug(f"{p.gradient_steps_per_iter/gradient_step_stopwatch.stop()} gradient steps/s")

        writer.add_scalar('Avg. Loss/train', avg_loss, i)
        writer.add_scalar('Avg. Reward/train', avg_rwd, i)
        writer.add_scalar('Win Ratio/train', win_ratio, i)
        writer.add_scalar('Avg. Steps/train', avg_steps, i)
        writer.add_scalars('Normalized #Actions_1/train', {
            '#Stop': act_counts[0][Action.Stop.value],
            '#Up': act_counts[0][Action.Up.value],
            '#Down': act_counts[0][Action.Down.value],
            '#Left': act_counts[0][Action.Left.value],
            '#Right': act_counts[0][Action.Right.value],
            '#Bomb': act_counts[0][Action.Bomb.value]
        }, i)
        writer.add_scalars('Normalized #Actions_2/train', {
            '#Stop': act_counts[1][Action.Stop.value],
            '#Up': act_counts[1][Action.Up.value],
            '#Down': act_counts[1][Action.Down.value],
            '#Left': act_counts[1][Action.Left.value],
            '#Right': act_counts[1][Action.Right.value],
            '#Bomb': act_counts[1][Action.Bomb.value]
        }, i)
        logging.debug(f"Iteration {i+1}/{p.num_iterations} finished after {iteration_stopwatch.stop()}s")

        # Do intermediate tests and save models
        logging.info("------------------------")
        if i % p.intermediate_test == p.intermediate_test-1:
            test_stopwatch=Stopwatch(start=True)
            logging.info("Testing model")        
            algo1.set_train(False)
            algo2.set_train(False)
            policy1 = algo1.get_policy()
            policy2 = algo2.get_policy()          
            model_save_path = log_dir + "/" + str(i)
            torch.save(algo1.q_network.state_dict(), model_save_path + '_1')
            torch.save(algo2.q_network.state_dict(), model_save_path + '_2')
            logging.info("Saved model to: " + model_save_path)
            data_generator.generate(p.episodes_per_eval, policy1, policy2, q1.get_transformer(), render=p.render_tests)
            algo1.set_train(True)
            algo2.set_train(True)
            logging.debug(f"Test finished after {test_stopwatch.stop()}s")
            logging.info("------------------------")
    writer.close()

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

def main(args):
    parser = argparse.ArgumentParser(description='Pommerman agent training script')
    parser.add_argument('-l', '--loglevel', type=str, dest="loglevel", default="INFO", help=f"Minimum loglevel to display. One of {[name for name in logging._nameToLevel]}")
    parser.add_argument('-i', '--iterations', type=int, dest="iterations", default=p.num_iterations, help=f"Number of iterations the model will be trained")

    args=parser.parse_args(args[1:])
    setup_logger(log_level=logging.getLevelName(args.loglevel))

    p.num_iterations=args.iterations

    test_pommerman_dqn()

# Only run main() if script if executed explicitly
if __name__ == '__main__':
    main(sys.argv)
