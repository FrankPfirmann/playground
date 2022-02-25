import numpy as np

class BombTracker:
    ''' keeps track of bomb position, life and blast strength for each agent'''
    def __init__(self, agents_n=4):
        self.agents_n = agents_n
        bomb_tracker = []
        for i in range(self.agents_n):
            bomb_tracker += [[]]
        self.bomb_tracker = bomb_tracker

    def step(self, obs, nobs):
        ''' update bomb tracker '''
        # reduce bomb life in bomb tracker
        for i in range(self.agents_n):
            for bomb in self.bomb_tracker[i]:
                bomb['life'] -= 1
        # remove bomb from tracker, if life reaches -3 (when flame dissappears)
        for i in range(self.agents_n):
            for j, bomb in enumerate(self.bomb_tracker[i]):
                if bomb['life'] <= -3:
                    del self.bomb_tracker[i][j]
        # add bomb to bombtracker if new bomb spawned
        for i in range(self.agents_n):
            agt_pos = obs[i]['position']
            if obs[i]['bomb_life'][agt_pos] == 0 and nobs[i]['bomb_life'][agt_pos] != 0: # check if agt succesfully layed a bomb
                self.bomb_tracker[i] += [{'position': agt_pos, 'life': 9, 'blast_strength': obs[i]['blast_strength']}]

    def get_killers(self, obs, nobs):
        ''' checks for killed agents and determines which other agent(s) killed it'''
        # determines all killed agents
        killed_agents = []
        for agent_id in obs[0]['alive']:
            if agent_id not in nobs[0]['alive']:
                killed_agents += [agent_id]
        killed_agents = list(map(lambda x: x - 10, killed_agents)) # get indices of agents instead of ids
        # check for all bombs that could have killed agent (do not take into account obstacles between agent and bomb)
        killers = []
        for killed_agent in killed_agents:
            possible_killers = []
            for killer, bombs in enumerate(self.bomb_tracker): 
                for bomb in bombs:
                    bomb_position = np.array(bomb['position'])
                    killed_agent_position = np.array(obs[killed_agent]['position'])
                    if np.isin(0, bomb_position - killed_agent_position): # check if bomb and agent are aligned
                        if np.all(np.abs(bomb_position-killed_agent_position) < bomb['blast_strength']): # check if bomb flame can reach agent
                            possible_killers += [killer]
            killers += [{'killed_agent': killed_agent, 'killers': possible_killers}]
            # TODO: account for obstacles
            # TODO: account for kicked bombs
        return killers


        

        


