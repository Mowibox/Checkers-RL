"""
@file        TDLambda_LVFA.py
@author      Mowibox (Ousmane THIONGANE)
@brief       TD(λ) Linear Value function Approximation algorithm 
@version     1.0
@date        2025-01-22
"""

# Imports
import pickle
import random
import sklearn
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

class VanillaFeatureEncoder:
    def __init__(self, env):
        self.env = env
    
    def encode(self, state):
        return np.array(state).flatten()
    
    @property
    def size(self):
        return len(self.encode(self.env.reset()[0]))
    

class RBFFeatureEncoder:
    def __init__(self, env, n_components=500):
        self.env = env
        self.observation_examples = np.array([np.array(env.reset()[0]).flatten() for _ in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(self.observation_examples)

        self.encoder = sklearn.pipeline.FeatureUnion([
           ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
           ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
           ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
           ("rbf4", RBFSampler(gamma=0.5, n_components=n_components)),
        ])
        self.features_obs = self.encoder.fit_transform(self.scaler.transform(self.observation_examples))
    
    def encode(self, state):
        scaled_state = self.scaler.transform([np.array(state).flatten()])
        return self.encoder.transform(scaled_state).reshape(-1)

    @property
    def size(self):
        return self.features_obs.shape[1]
    
class TDLambda_LVFA:
    def __init__(self, env, feature_encoder_cls=RBFFeatureEncoder, alpha=0.01, alpha_decay=1,
                 gamma=0.9999, epsilon=0.3, epsilon_decay=0.995, final_epsilon=0.2, lambda_=0.9):
        self.env = env
        self.feature_encoder = feature_encoder_cls(env)
        self.shape = (len(env.available_moves()), self.feature_encoder.size)
        self.weights = np.random.random(self.shape)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.lambda_ = lambda_

    def Q(self, feats):
        return self.weights@feats.reshape(-1, 1)
    
    def update_transition(self, s, action, s_prime, reward, available_moves, done):
        s_feats = self.feature_encoder.encode(s)
        s_prime_feats = self.feature_encoder.encode(s_prime)
        action_idx = available_moves.index(action)
        print(20*"=")
        print(f"Action Index: {action_idx}, Weights Shape: {self.weights.shape}")
        print(f"Available Moves: {available_moves}")
        print(f"Action: {available_moves[action_idx]}")


        delta = reward + (1-done)*self.gamma*np.max(self.Q(s_prime_feats)) - self.Q(s_feats)[action_idx]
        self.traces = self.gamma*self.lambda_*self.traces
        self.traces[action_idx] += s_feats
        self.weights[action_idx] -= self.alpha*delta*self.traces[action_idx]
    
    def update_alpha_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.final_epsilon*self.epsilon_decay)
        self.alpha = self.alpha = self.alpha*self.alpha_decay

    def policy(self, state):
        state_feats = self.feature_encoder.encode(state)
        best_action_idx = self.Q(state_feats).argmax()
        return self.env.available_moves()[best_action_idx]
    
    def epsilon_greedy(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        available_moves = self.env.available_moves()
        if random.random() < epsilon:
            return random.choice(available_moves)
        return self.policy(state)

    def train(self, episodes=200):
        print(f'ep | eval | epsilon | alpha')
        for episode in range(episodes):
            state, _ = self.env.reset()
            done = False
            self.traces = np.zeros(self.shape)
            while not done:
                available_moves = self.env.available_moves()
                action = self.epsilon_greedy(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_transition(state, action, next_state, reward, available_moves, done)
                state = next_state

        self.update_alpha_epsilon()
        print(episode, self.evaluate(), self.epsilon, self.alpha)

    def evaluate(self, episodes=10):
        rewards = []
        for _ in range(episodes):
            done = False 
            state, _ = self.env.reset()
            total_reward = 0
            while not done:
                action = self.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
            rewards.append(total_reward)
        return np.mean(rewards)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        return pickle.load(open(filename,'rb'))