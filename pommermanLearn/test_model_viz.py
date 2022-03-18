import torch

from data_generator import DataGeneratorPommerman
from dqn import DQN
from models.pommer_q import Pommer_Q

num_its = 10
episodes_per_iter = 10

def test_pommerman_dqn(model_dir):
	q = Pommer_Q()
	q_target = Pommer_Q()

	q.load_state_dict(torch.load(model_dir))
	q_target.load_state_dict(torch.load(model_dir))

	algo = DQN(q, q_target)
	data_generator = DataGeneratorPommerman()


	for i in range(num_its):
		print("Iteration: " + str(i))
		policy = algo.get_policy()

		res, ties, avg_rwd = data_generator.generate(episodes_per_iter, policy)
		win_ratio = res[0] / (sum(res)+ties)

		algo.set_train(False)
		policy = algo.get_policy()

		data_generator.generate(episodes_per_iter, policy, render=True)
	

test_pommerman_dqn("./data/tensorboard/20211208T132146/1419")