# main.py

num_iterations = 1000000
episodes_per_iter = 1
gradient_steps_per_iter = 50
batch_size = 16
episodes_per_eval = 10
intermediate_test = 1
render_tests = False
board_size=8

#dqn.py

seed = 1

gamma = 0.99
tau = 0.005
hidden_size = 256
lr_q = 0.0003
lr_policy = 0.0003

max_trans = 100000
warmup_trans = 500
eval_every = 500
eval_episodes = 10
exploration_noise = 0.1

#data_generator.py

replay_size = 1000000
max_steps = 500
reward_func = 'BombReward'
