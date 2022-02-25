from datetime import datetime
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
env = 'PommeRadio-v2'  # PommeFFACompetition-v0 or OneVsOne-v0 or PommeTeamCompetition-v0
p_observable = True
crop_fog=False
double_q = True
prioritized_replay = True
beta = 0 # determines how replays should be weighted (beta==0 --> all weights are 1, beta==1 --> influence of replays is fully normalized)
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
run_name = datetime.now().strftime("%Y%m%dT%H%M%S")

#train_agent.py
communicate=True
use_memory=True
#dqn.py
seed = 1

gamma = 0.99
tau = 0.005
lr_q = 0.0003

exploration_noise = 0.0
exploration_dropoff = 0.01
explortation_min = 0.0
#data_generator.py
set_position=False
replay_size = 2**16 # must be a power of 2 to be compatible with prioritized replay
max_steps = 800
reward_func = 'SkynetReward' #SkynetReward, BombReward

#models.py
memory_method = 'forgetting' # one of 'counting', 'forgetting'
forgetfullness=0.05

def validate():
    if use_memory: assert p_observable and not crop_fog
    if communicate: assert use_memory