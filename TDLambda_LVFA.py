"""
@file        TDLambda_LVFA.py
@author      Mowibox (Ousmane THIONGANE)
@brief       TD(λ) Linear Value Function Approximation algorithm 
@version     1.0
@date        2025-01-22
"""
# Imports
import pickle
import random
import sklearn
import numpy as np
import sklearn.pipeline
from copy import deepcopy
import sklearn.preprocessing
from CheckersRLFeaturesEncoder import CheckersRLFeaturesEncoder
    

class TDLambda_LVFA:
    """
    TD(λ) Linear Value function Approximation class
    """
    def __init__(self, env, feature_encoder_cls=CheckersRLFeaturesEncoder, alpha=0.01, alpha_decay=1,
                 gamma=0.9999, epsilon=0.3, epsilon_decay=0.995, final_epsilon=0.2, lambda_=0.9):
        """
        Initializes the TD(λ) Linear Value Function Approximation

        @param env: The provided environment
        @param feature_encoder_cls: The feature encoder class
        @param alpha: The learning rate
        @param alpha_decay: The learning rate decay
        @param gamma: The discount factor
        @param epsilon: The exploration rate
        @param epsilon_decay: The epsilon decay
        @param final_epsilon: The minimum value epsilon can decay to
        @param lambda: The TD(λ) trace decay parameter
        """
        self.env = env
        self.feature_encoder = feature_encoder_cls(env)
        self.weights = np.random.random(self.feature_encoder.size)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.lambda_ = lambda_
        self.traces = np.zeros_like(self.weights)
        self.agent_mark = None

    def V(self, state: list) -> np.ndarray:
        """
        Computes the estimated value of a given state

        @param state: The current state
        """
        feats = self.feature_encoder.encode(state, self.agent_mark)
        return np.dot(self.weights, feats)
    
    def simulate_action(self, state: int, action: list[tuple], player: tuple) -> tuple[list, int]:
        """
        Simulates an action 

        @param state: The current state
        @param action: The chosen action
        @param player: The current player
        """
        new_state = deepcopy(state)
        new_state, new_player = self.env.transition_function(new_state, action, player, simulation=True)
        return new_state, new_player
    
    def update_transition(self, s: list, action: list[tuple], s_prime: int, 
                          reward: float, done: bool, current_player: int, next_player: int):
        """
        Updates the Value function after a transition

        @param s: The current state
        @param action: The chosen action
        @param s_prime: The next state
        @param reward: The reward
        @param done: The episode ending flag
        @param current_player: The current player
        @param next_player: The next player
        """
        s_feats = self.feature_encoder.encode(s, self.agent_mark)
        Vs = np.dot(self.weights, s_feats)

        if next_player == self.agent_mark:
            Vs_prime = np.dot(self.weights, self.feature_encoder.encode(s_prime, self.agent_mark))
        else:
            Vs_prime = -np.dot(self.weights, self.feature_encoder.encode(s_prime, self.agent_mark))

        delta = reward + (1-done)*self.gamma*Vs_prime - Vs
        self.traces = self.gamma*self.lambda_*self.traces + s_feats
        self.weights += self.alpha*delta*self.traces
    
    def update_alpha_epsilon(self):
        """
        Updates the learning rate and exploration rate values
        """
        self.epsilon = max(self.final_epsilon, self.epsilon*self.epsilon_decay)
        self.alpha *= self.alpha_decay

    def policy(self, state: list, current_player: int) -> list[tuple]:
        """
        Selects the best action for a given state 

        @param state: The current state
        @param current_player: The current player
        """
        actions = self.env.available_moves(state, current_player)
        best_val = -float('inf')
        best_action = None
        for action in actions:
            next_state, next_player = self.simulate_action(state, action, current_player)
            if next_player == self.agent_mark:
                value = np.dot(self.weights, self.feature_encoder.encode(next_state, self.agent_mark))
            else:
                value = -np.dot(self.weights, self.feature_encoder.encode(next_state, self.agent_mark))
            if value > best_val:
                best_val = value
                best_action = action
        return best_action
    
    def epsilon_greedy(self, state: list, current_player: int, epsilon: float=None):
        """
        Computes the epsilon-greedy policy

        @param state: The current state
        @param current_player: The current player
        @param epsilon: The exploration rate
        """
        if epsilon is None:
            epsilon = self.epsilon
        actions = self.env.available_moves(state, current_player)
        if current_player != self.agent_mark:
            return random.choice(actions)
        if random.random() < epsilon:
            return random.choice(actions)
        return self.policy(state, current_player)


    def train(self, episodes: int=1000):
        """
        Training function

        @param episodes: The number of episodes
        """
        print(f'ep | eval | epsilon | alpha')
        for episode in range(episodes):
            state, current_player = self.env.reset()
            if self.agent_mark is None:
                self.agent_mark = current_player
            done = False
            self.traces = np.zeros_like(self.weights)
            while not done:
                action = self.epsilon_greedy(state, current_player)
                if action is None:
                    break
                next_state, reward, done, next_player = self.env.step(action)
                self.update_transition(state, action, next_state, reward, done, current_player, next_player)
                state = next_state
                current_player = next_player

            self.update_alpha_epsilon()
            if episode % 20 == 0:
                print(episode, self.evaluate(), self.epsilon, self.alpha)

    def evaluate(self, episodes: int=10) -> float:
        """
        Evaluation function

        @param episodes: The number of epsiodes
        """
        rewards = []
        for _ in range(episodes):
            state, current_player = self.env.reset()
            total_reward = 0
            done = False 
            if self.agent_mark is None:
                self.agent_mark = current_player
            while not done:
                action = self.policy(state, current_player)
                if action is None:
                    break
                next_state, reward, done, next_player = self.env.step(action)
                total_reward += reward
                current_player = next_player
                state = next_state
            rewards.append(total_reward)
        return np.mean(rewards)

    def save(self, filename: str):
        """
        Saves the model with the provided filename

        @param filename: The filename
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str):
        """
        Loads a model

        @param filename: The filename
        """
        return pickle.load(open(filename,'rb'))