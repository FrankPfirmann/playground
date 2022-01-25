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
import params as p
from pommerman.constants import Action

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
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

    if p.env == 'OneVsOne-v0':
        board_size = 8
    else:
        board_size = 11
    obs_size = board_size if not p.centralize_planes else board_size*2-1
    if p.p_observable:
        transform_func = transform_observation_partial
    elif p.centralize_planes:
        transform_func = transform_observation_centralized
    else:
        transform_func = transform_observation_simple

    q = Pommer_Q(p.p_observable or p.centralize_planes, transform_func)
    q.to(device)
    torch.manual_seed(p.seed)
    q_target = Pommer_Q(p.p_observable or p.centralize_planes, transform_func)
    q_target.to(device)
    algo = DQN(q, q_target, p.exploration_noise, device=device, dq=p.double_q)
    data_generator = DataGeneratorPommerman(
	p.env,
	augmenter=[
            DataAugmentor_v1()
        ])
    run_name=datetime.now().strftime("%Y%m%dT%H%M%S")
    log_dir=os.path.join("./data/tensorboard/", run_name)
    logging.info(f"Staring run {run_name}")
    writer = SummaryWriter(log_dir=log_dir)
    backsize = 1
    backplay_interval = int(np.floor(p.num_iterations/1000))
    explo = p.exploration_noise
    for i in range(p.num_iterations):
        logging.info(f"Iteration {i+1}/{p.num_iterations} started")
        iteration_stopwatch = Stopwatch(start=True)
        policy = algo.get_policy()

        res, ties, avg_rwd, act_counts, avg_steps = data_generator.generate(p.episodes_per_iter, policy,
                                                                  q.get_transformer(), 'train', 'static:0')
        explo = max(p.explortation_min, explo - p.exploration_dropoff)
        algo.set_exploration(explo)
        act_counts=[act/sum(act_counts) for act in act_counts] # Normalize
        #The agents wins are stored at index 0 i the data_generator
        win_ratio = res[0] / (sum(res)+ties)

        total_loss=0
        gradient_step_stopwatch=Stopwatch(start=True)
        #increase backplay range (stop at
        if p.backplay:
            if(i + 1) % backplay_interval == 0:
                backsize += 1

        for j in range(p.gradient_steps_per_iter):
            if p.backplay:
                batch_s = min(len(data_generator.episode_buffer) * backsize, p.batch_size)
                batch = data_generator.get_batch_buffer_back(batch_s, backsize)
            elif p.episode_backward:
                batch = data_generator.get_episode_buffer()
            else:
                batch = data_generator.get_batch_buffer(p.batch_size)
            loss=algo.train(batch)
            total_loss+=loss
        avg_loss=total_loss/p.gradient_steps_per_iter
        logging.debug(f"{p.gradient_steps_per_iter/gradient_step_stopwatch.stop()} gradient steps/s")

        writer.add_scalar('Avg. Loss/train', avg_loss, i)
        writer.add_scalar('Avg. Reward/train', avg_rwd, i)
        writer.add_scalar('Win Ratio/train', win_ratio, i)
        writer.add_scalar('Avg. Steps/train', avg_steps, i)
        writer.add_scalars('Normalized #Actions/train', {
            '#Stop': act_counts[Action.Stop.value],
            '#Up': act_counts[Action.Up.value],
            '#Down': act_counts[Action.Down.value],
            '#Left': act_counts[Action.Left.value],
            '#Right': act_counts[Action.Right.value],
            '#Bomb': act_counts[Action.Bomb.value]
        }, i)
        logging.debug(f"Iteration {i+1}/{p.num_iterations} finished after {iteration_stopwatch.stop()}s")

        logging.info("------------------------")
        if i % p.intermediate_test == p.intermediate_test-1:
            test_stopwatch=Stopwatch(start=True)
            logging.info("Testing model")
            
            algo.set_train(True)
            policy = algo.get_policy()
            
            model_save_path = log_dir + "/" + str(i)
            torch.save(algo.q_network.state_dict(), model_save_path)
            logging.info("Saved model to: " + model_save_path)
            data_generator.generate(p.episodes_per_eval, policy, q.get_transformer(), 'train', 'static:0', render=p.render_tests)
            algo.set_train(True)
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
