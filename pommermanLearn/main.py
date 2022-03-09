#!/usr/bin/env python3
# solve error #15
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
if os.getuid() == 1000:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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
from models import PommerQEmbeddingMLP
from models import PommerQEmbeddingRNN
from embeddings import PommerLinearAutoencoder
from embeddings import PommerConvAutoencoder
from util.analytics import Stopwatch
from util.data import transform_observation_simple, transform_observation_partial, transform_observation_centralized, transform_observation_partial_uncropped


def set_all_seeds():
    torch.manual_seed(p.seed)
    np.random.seed(p.seed)
    random.seed(p.seed)


def init_result_dict(length):
    res = {"loss": np.zeros(length),
           "reward": np.zeros(length),
           "win_ratio": np.zeros(length),
           "steps": np.zeros(length)}
    return res


def train_dqn(dqn1=None, dqn2=None, num_iterations=p.num_iterations, episodes_per_iter=p.episodes_per_iter,
              augmentors=[], enemy='static:0', max_steps=p.max_steps, mean_run=False):

    if p.env == 'OneVsOne-v0':
        board_size = 8
    else:
        board_size = 11
    get_transform_func()
    # LSTM model with object embeddings:
    """
    path="./data/models/PommerConvAutoEncoder-64-db16e0d.pth"
    embedding_model = PommerConvAutoencoder(embedding_dims=64)
    embedding_model.load_state_dict(torch.load(path))
    embedding_model.eval() 
    embedding_model.mode='encode'
    q_target = PommerQEmbeddingRNN(embedding_model)
    q = PommerQEmbeddingRNN(embedding_model)
    """

    # initialize DQN and data generator
    if dqn1 is None:
        dqn1 = create_dqn()

    if dqn2 is None:
        dqn2 = create_dqn()

    data_generator = DataGeneratorPommerman(
        p.env,
        augmentors=augmentors
    )

    # start logging
    if mean_run:
        results_dict = init_result_dict(num_iterations)
    else:
        results_dict = None
        log_dir = os.path.join("./data/tensorboard/", p.run_name)
        logging.info(f"Staring run {p.run_name}")
        writer = SummaryWriter(log_dir=log_dir)
    exploration = p.exploration_noise
    # training loop

    test_i = 0
    for i in range(num_iterations):
        logging.info(f"Iteration {i + 1}/{num_iterations} started")
        iteration_stopwatch = Stopwatch(start=True)
        policy1 = dqn1.get_policy()
        policy2 = dqn2.get_policy()

        # generate data an store normalized act counts and win ration
        res, ties, avg_rwd, act_counts, avg_steps = data_generator.generate(episodes_per_iter, policy1, policy2, enemy,
                                                                            dqn1.q_network.get_transformer(), 'train',
                                                                            'train', max_steps)
        act_counts[0] = [act / max(1, sum(act_counts[0])) for act in act_counts[0]]
        act_counts[1] = [act / max(1, sum(act_counts[1])) for act in act_counts[1]]

        dqn1.set_exploration(exploration)
        dqn2.set_exploration(exploration)
        # The agents wins are stored at index 0 i the data_generator
        win_ratio = res[0] / (sum(res) + ties)

        # fit models on generated data
        total_loss = 0
        gradient_step_stopwatch = Stopwatch(start=True)

        def train_episode(dqn, ind):
            added_loss = 0
            for _ in range(p.gradient_steps_per_iter):
                batch = data_generator.get_batch_buffer(p.batch_size, ind)
                if p.use_nstep:
                    batch_n = data_generator.get_batch_buffer_from_idx(batch[6], ind)
                    loss, indexes, td_error = dqn.train(batch, batch_n)
                else:
                    loss, indexes, td_error = dqn.train(batch)
                added_loss += loss
                if p.prioritized_replay:
                    data_generator.update_priorities(indexes, td_error, ind)
                dqn.reset_net_noise()
            return added_loss

        total_loss += train_episode(dqn1, 0)
        total_loss += train_episode(dqn2, 1)
        avg_loss = total_loss / p.gradient_steps_per_iter

        # logging
        logging.debug(f"{p.gradient_steps_per_iter / gradient_step_stopwatch.stop()} gradient steps/s")
        if mean_run:
            results_dict["loss"][i] = avg_loss
            results_dict["reward"][i] = avg_rwd
            results_dict["win_ratio"][i] = win_ratio
            results_dict["steps"][i] = avg_steps
        else:
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
        logging.debug(f"Iteration {i + 1}/{num_iterations} finished after {iteration_stopwatch.stop()}s")

        # Do intermediate tests and save models
        logging.info("------------------------")
        if i % p.intermediate_test == p.intermediate_test - 1:
            test_stopwatch = Stopwatch(start=True)
            logging.info("Testing model")
            dqn1.set_train(False)
            dqn2.set_train(False)
            policy1 = dqn1.get_policy()
            policy2 = dqn2.get_policy()
            if not mean_run:
                model_save_path = log_dir + "/" + str(i)
                torch.save(dqn1.q_network.state_dict(), model_save_path + '_1')
                torch.save(dqn2.q_network.state_dict(), model_save_path + '_2')
                logging.info("Saved model to: " + model_save_path)
            test_res, test_ties, test_avg_rwd, test_act_counts, test_avg_steps = data_generator.generate(p.episodes_per_eval, policy1, policy2, enemy, 
                                                                                dqn1.q_network.get_transformer(), 'train', 
                                                                                'train', max_steps, render=p.render_tests, test=True)
            
            test_win_ratio = test_res[0] / (sum(test_res) + test_ties)

            writer.add_scalar('Avg. Test Reward/train', test_avg_rwd, test_i)
            writer.add_scalar('Test Win Ratio/train', test_win_ratio, test_i)
            writer.add_scalar('Avg. Test Steps/train', test_avg_steps, test_i)

            dqn1.set_train(True)
            dqn2.set_train(True)
            logging.debug(f"Test finished after {test_stopwatch.stop()}s")
            logging.info("------------------------")
            test_i += 1

    if not mean_run:
        writer.close()
    return dqn1, dqn2, results_dict


def get_transform_func():
    if p.p_observable and not p.use_memory:
        transform_func = transform_observation_partial
    elif p.p_observable and p.use_memory and not p.crop_fog:
        transform_func = transform_observation_partial_uncropped
    elif p.centralize_planes:
        transform_func = transform_observation_centralized
    else:
        transform_func = transform_observation_simple
    return transform_func


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


def create_dqn(path=None):
    transform_func = get_transform_func()
    support = torch.linspace(p.v_min, p.v_max, p.atom_size).to(p.device)
    q1 = Pommer_Q(p.p_observable or p.centralize_planes, transform_func, support=support)
    q_target1 = Pommer_Q(p.p_observable or p.centralize_planes, transform_func, support=support)
    q1.to(p.device)
    q_target1.to(p.device)

    model_dir = path
    if path is not None:
        q1.load_state_dict(torch.load(model_dir, map_location=p.device))
        q_target1.load_state_dict(torch.load(model_dir, map_location=p.device))
    dqn1 = DQN(q1, q_target1, p.exploration_noise, dq=p.double_q, device=p.device, support=support)
    return dqn1


def main(args):
    parser = argparse.ArgumentParser(description='Pommerman agent training script')
    parser.add_argument('-l', '--loglevel', type=str, dest="loglevel", default="INFO",
                        help=f"Minimum loglevel to display. One of {[name for name in logging._nameToLevel]}")
    parser.add_argument('-i', '--iterations', type=int, dest="iterations", default=p.num_iterations,
                        help=f"Number of iterations the model will be trained")
    if p.seed != -1:
        set_all_seeds()
    parser.add_argument('-e', '--episodes', type=int, dest="episodes", default=p.episodes_per_iter,
                        help=f"Number of episodes played per iteration")
    parser.add_argument('-n', '--name', type=str, dest="name", default=p.run_name,
                        help=f"Name of the run")

    args = parser.parse_args(args[1:])
    setup_logger(log_level=logging.getLevelName(args.loglevel))

    p.num_iterations    = args.iterations
    p.episodes_per_iter = args.episodes
    p.run_name          = args.name
    if p.seed != -1:
        set_all_seeds()
    #do_mean_run(1, 100)
    #dqn1 = create_dqn("data/tensorboard/long/20220306T192354/2699_1")
    #dqn2 = create_dqn("data/tensorboard/long/20220306T192354/2699_2")
    dqn1, dqn2 = None, None

    p.validate()

    dqn1, dqn2, _ = train_dqn(dqn1, dqn2, num_iterations=10000, enemy='static:0', augmentors=[])


def do_mean_run(mean_run_n, mean_num_iterations):
    mean_dict = init_result_dict(mean_num_iterations)
    for i in range(0, mean_run_n):
        _, _, res = train_dqn(num_iterations=mean_num_iterations, enemy='static:0', augmentors=[], mean_run=True)
        for key in mean_dict.keys():
            mean_dict[key] += res[key] / mean_run_n
    log_dir = os.path.join("./data/tensorboard/mean/", p.run_name)
    logging.info(f"Writing mean run {p.run_name}")
    writer = SummaryWriter(log_dir=log_dir)
    for i in range(0, mean_num_iterations):
        writer.add_scalar('Avg. Loss/train', mean_dict["loss"][i], i)
        writer.add_scalar('Avg. Reward/train', mean_dict["reward"][i], i)
        writer.add_scalar('Win Ratio/train', mean_dict["win_ratio"][i], i)
        writer.add_scalar('Avg. Steps/train', mean_dict["steps"][i], i)
    writer.close()


# Only run main() if script if executed explicitly
if __name__ == '__main__':
    main(sys.argv)
