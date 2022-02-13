import pommerman as pm
from pommerman.agents.http_agent import HttpAgent
from pommerman.agents.simple_agent import SimpleAgent

remote_agent_1 = HttpAgent(
    port=10080,
    host='localhost'
)

remote_agent_2 = HttpAgent(
    port=10081,
    host='localhost'
)

agent_list=[
    remote_agent_1,
    SimpleAgent(),
    remote_agent_2,
    SimpleAgent()
]
env = pm.make("PommeRadioCompetition-v2", agent_list)

obs=env.reset()
done=False

while not done:
    act=env.act(obs)
    obs, rwd, done, _ = env.step(act)
    print(f"Actions: {act}")
print(f"Result: {rwd}")
env.close()