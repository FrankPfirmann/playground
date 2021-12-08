
def staying_alive_reward(nobs, agent_id):
    """
    Return a reward if the agent with the given id is alive.

    :param nobs: The game state
    :param agent_id: The agent to check

    :return: The reward for staying alive
    """
    #print(nobs[0]['position'][0])
    if agent_id in nobs[0]['alive']:
        return 1.0
    else:
        return 0.0

def go_right_reward(nobs, obs, agent_num):
    """
    Return a reward for going to the right side of the board

    :param nobs: The current observation
    :param obs: The last observation
    :param agent_num: The id of the agent to check

    :return: The reward for going right
    """
    return nobs[agent_num]['position'][0] - obs[agent_num]['position'][0]

