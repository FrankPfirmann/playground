import torch

# main.py
num_iterations = 1000
episodes_per_iter = 1
gradient_steps_per_iter = 100
batch_size = 128
episodes_per_eval = 30
intermediate_test = 100
centralize_planes = True
render_tests = False
env = 'PommeTeamCompetition-v0'  # PommeFFACompetition-v0 or OneVsOne-v0 or PommeTeamCompetition-v0
p_observable = True
crop_fog=True
double_q = True
prioritized_replay = True
beta = 0 # determines how replays should be weighted (beta==0 --> all weights are 1, beta==1 --> influence of replays is fully normalized)
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
communicate = False
#dqn.py
seed = 1

gamma = 0.99
tau = 0.005
lr_q = 0.0003

exploration_noise = 0.1
exploration_dropoff = 0.01
explortation_min = 0.05
#data_generator.py

replay_size = 2**16 # must be a power of 2 to be compatible with prioritized replay
max_steps = 800
reward_func = 'SkynetReward' #SkynetReward, BombReward

#models.py
use_memory=False
memory_method = 'forgetting' # one of 'counting', 'forgetting'
forgetfullness=0.05

def validate():
    if use_memory: assert p_observable and not crop_fog