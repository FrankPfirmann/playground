from pommerman.agents import BaseAgent
from pommerman.runner import DockerAgentRunner

from pommermanLearn.agents.dqn_agent import DQNAgent

class DockerAgent(DQNAgent, DockerAgentRunner):
    """
    An agent that exposes a REST API for usage in a docker container.
    """
    def __init__(self, model_dir):
        super(DockerAgent, self).__init__(model_dir)