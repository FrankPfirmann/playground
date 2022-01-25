# main.py

num_iterations = 1000000
episodes_per_iter = 1
gradient_steps_per_iter = 10
batch_size = 16
episodes_per_eval = 10
intermediate_test = 10
centralize_planes = True
render_tests = False
env = 'PommeTeamCompetition-v0'  # PommeFFACompetition-v0 or OneVsOne-v0 or PommeTeamCompetition-v0
episode_backward = False
p_observable = True
backplay = False
double_q = True

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
exploration_dropoff = 0.01
explortation_min = 0.05
#data_generator.py

replay_size = 50000
max_steps = 800
reward_func = 'SkynetReward' #SkynetReward, BombReward
