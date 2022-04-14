import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, eps_start=1.0, eps_min=0.00000001, eps_decay=0.9995, alpha=0.6, gamma=1.0, nA=6):
        """ Initialize agent.
        ### epsilon-greedy policy is controlled through an epsilon start, an epsilon min and a decay rate, and is updated during the step method ###

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.alpha = alpha

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        probs = np.ones(self.nA)*self.epsilon/self.nA
        best_a_index = np.argmax(self.Q[state])
        probs[best_a_index] = 1 - self.epsilon + self.epsilon / self.nA
        action = np.random.choice(np.arange(self.nA), p=probs)
        
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        probs = np.ones(self.nA)*self.epsilon/self.nA
        best_a_index = np.argmax(self.Q[next_state])
        probs[best_a_index] = 1 - self.epsilon + self.epsilon / self.nA
        
        old_Q = self.Q[state][action]
        ### Update Q using Expected SARSA
        self.Q[state][action] = old_Q + self.alpha*(reward + self.gamma*(np.dot(self.Q[next_state], probs)) - old_Q)
        ### Update Q using SARSAMAX / Q Learning
        #self.Q[state][action] = old_Q + self.alpha*(reward + self.gamma*(np.max(self.Q[next_state])) - old_Q)
        ### Update epsilon of the epsilon-greedy policy
        self.epsilon = max(self.epsilon*self.eps_decay, self.eps_min)
