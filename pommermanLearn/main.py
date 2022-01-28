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

def train_dqn(dqn1=None, dqn2=None, num_iterations=p.num_iterations, episodes_per_iter=p.episodes_per_iter, augmentors=[], enemy='static:0', max_steps=p.max_steps) :
    torch.manual_seed(p.seed)
    np.random.seed(p.seed)
    random.seed(p.seed)
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

    if p.p_observable:
        transform_func = transform_observation_partial
    elif p.centralize_planes:
        transform_func = transform_observation_centralized
    else:
        transform_func = transform_observation_simple

    # initialize DQN and data generator
    if dqn1 == None:
        q1 = Pommer_Q(p.p_observable or p.centralize_planes, transform_func)
        q_target1 = Pommer_Q(p.p_observable or p.centralize_planes, transform_func)
        q1.to(device)
        q_target1.to(device)
        dqn1 = DQN(q1, q_target1, p.exploration_noise)

    if dqn2 == None:
        q2 = Pommer_Q(p.p_observable, transform_func)
        q_target2 = Pommer_Q(p.p_observable, transform_func)
        q2.to(device)
        q_target2.to(device)
        dqn2 = DQN(q2, q_target2, p.exploration_noise)

    data_generator = DataGeneratorPommerman(
	    p.env,
	    augmentors=augmentors
    )

    # start logging
    run_name=datetime.now().strftime("%Y%m%dT%H%M%S")
    log_dir=os.path.join("./data/tensorboard/", run_name)
    logging.info(f"Staring run {run_name}")
    writer = SummaryWriter(log_dir=log_dir)
    backsize = 1
    backplay_interval = int(np.floor(p.num_iterations/1000))
    explo = p.exploration_noise
    # training loop
    for i in range(num_iterations):
        logging.info(f"Iteration {i+1}/{num_iterations} started")
        iteration_stopwatch = Stopwatch(start=True)
        policy1 = dqn1.get_policy()
        policy2 = dqn2.get_policy()

        # generate data an store normalized act counts and win ration
        res, ties, avg_rwd, act_counts, avg_steps = data_generator.generate(episodes_per_iter, policy1, policy2, enemy, dqn1.q_network.get_transformer(), 'train', 'train', max_steps)
        act_counts[0] = [act/sum(act_counts[0]) for act in act_counts[0]]
        act_counts[1] = [act/sum(act_counts[1]) for act in act_counts[1]]

        explo = max(p.explortation_min, explo - p.exploration_dropoff)
        dqn1.set_exploration(explo)
        dqn2.set_exploration(explo)
        #The agents wins are stored at index 0 i the data_generator
        win_ratio = res[0] / (sum(res)+ties)

        # fit models on generated data
        total_loss=0
        gradient_step_stopwatch=Stopwatch(start=True)
        #increase backplay range (stop at
        if p.backplay:
            if(i + 1) % backplay_interval == 0:
                backsize += 1
        for _ in range(p.gradient_steps_per_iter):
            if p.backplay:
                batch_s = min(len(data_generator.episode_buffer) * backsize, p.batch_size)
                batch1 = data_generator.get_batch_buffer_back(batch_s, backsize)
            elif p.episode_backward:
                batch1 = data_generator.get_episode_buffer()
            else:
                batch1 = data_generator.get_batch_buffer(p.batch_size, 0)
            loss, indexes, td_error = dqn1.train(batch1)
            if p.prioritized_replay:
                data_generator.update_priorities(indexes, td_error, 0)
            total_loss += loss

        for _ in range(p.gradient_steps_per_iter):
            if p.backplay:
                batch_s = min(len(data_generator.episode_buffer) * backsize, p.batch_size)
                batch2 = data_generator.get_batch_buffer_back(batch_s, backsize)
            elif p.episode_backward:
                batch2 = data_generator.get_episode_buffer()
            else:
                batch2 = data_generator.get_batch_buffer(p.batch_size, 1)
            loss, indexes, td_error = dqn2.train(batch2)
            if p.prioritized_replay:
                data_generator.update_priorities(indexes, td_error, 1)
                total_loss += loss

        avg_loss=total_loss/p.gradient_steps_per_iter

        # logging
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
        logging.debug(f"Iteration {i+1}/{num_iterations} finished after {iteration_stopwatch.stop()}s")

        # Do intermediate tests and save models
        logging.info("------------------------")
        if i % p.intermediate_test == p.intermediate_test-1:
            test_stopwatch=Stopwatch(start=True)
            logging.info("Testing model")
            dqn1.set_train(False)
            dqn2.set_train(False)
            policy1 = dqn1.get_policy()
            policy2 = dqn2.get_policy()
            model_save_path = log_dir + "/" + str(i)
            torch.save(dqn1.q_network.state_dict(), model_save_path + '_1')
            torch.save(dqn2.q_network.state_dict(), model_save_path + '_2')
            logging.info("Saved model to: " + model_save_path)
            data_generator.generate(p.episodes_per_eval, policy1, policy2, enemy, dqn1.q_network.get_transformer(), 'train', 'train', max_steps, render=p.render_tests)
            dqn1.set_train(True)
            dqn2.set_train(True)
            logging.debug(f"Test finished after {test_stopwatch.stop()}s")
            logging.info("------------------------")
    writer.close()
    return dqn1 , dqn2

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

    train_dqn(num_iterations=2000, enemy='smart_random', augmentors=[])
    p.gradient_steps_per_iter = 200
    p.batch_size = 32
    train_dqn(num_iterations=2000, enemy='smart_random', augmentors=[DataAugmentor_v1()])

    # dqn1, dqn2 = train_dqn(dqn1=dqn1, dqn2=dqn2, num_iterations=10000, enemy='smart_random_no_bomb', augmentors=[])

# Only run main() if script if executed explicitly
if __name__ == '__main__':
    main(sys.argv)