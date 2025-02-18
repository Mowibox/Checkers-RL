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
from copy import deepcopy
import sklearn.preprocessing
from CheckersRL import CheckersRL


###############################################################################
# Feature Encoder
###############################################################################

class CheckersRLFeaturesEncoder:
    """
    """
    def __init__(self, env):
        self.env = env
        self.feature_sizes = {
            "PawnAdvantage": 4,
            "PawnDisadvantage": 4,
            "PawnThreat": 3,
            "PawnTake": 3,
            "Backrowbridge": 1,
            "CentreControl": 3,
            "XCentreControl": 3,
            "TotalMobility": 4,
            "Exposure": 3,
            "Advancement": 3,
            "DoubleDiagonal": 4,
            "DiagonalMoment": 3,
            "KingCentreControl": 3,
            "Threat": 3,
            "Taken": 3,
        }
        self.feature_size = sum(self.feature_sizes.values())
        self.n_pawns = sum(tile == self.env.WHITE_PAWN for row in self.env.board for tile in row)

    
    def encode(self, state, player=CheckersRL.WHITE_PAWN):
        """

        @param state:
        @param player: ()
        """
        features = []

        player_pawns = self.pawns_for(player)
        opponent_pawns = self.pawns_for(self.opponent(player))

        # F1: PawnAdvantage
        n_player = sum(1 for row in state for tile in row if tile in player_pawns)
        n_opponent = sum(1 for row in state for tile in row if tile in opponent_pawns)
        pawn_advantage = n_player - n_opponent
        features.append(self.normalize(pawn_advantage, -self.n_pawns, self.n_pawns, self.feature_sizes["PawnAdvantage"]))

        # F2: PawnDisadvantage
        features.append(self.normalize(-pawn_advantage, -self.n_pawns, self.n_pawns, self.feature_sizes["PawnDisadvantage"]))

        # F3: PawnThreat
        pawn_threat = self.threatened_pawns(state, player)
        features.append(self.normalize(pawn_threat, 0, self.n_pawns, self.feature_sizes["PawnThreat"]))

        # F4: PawnTake
        pawn_take = self.capture_moves(state, player)
        features.append(self.normalize(pawn_take, 0, self.n_pawns, self.feature_sizes["PawnTake"]))

        # F7: Backrowbridge
        if player == self.env.WHITE_PAWN:
            backrow = any(tile == self.env.WHITE_PAWN for tile in state[-1])
        else: 
            backrow = any(tile == self.env.BLACK_PAWN for tile in state[0])
        features.append(np.array([1])) if backrow else features.append(np.array([0]))

        # F8: Centrecontrol
        centre_control = self.center_pawns(state, player)
        features.append(self.normalize(centre_control, 0, self.n_pawns, self.feature_sizes["CentreControl"]))

        # F9: XCentrecontrol
        xcentre_control = self.xcenter_pawns(state, player)
        features.append(self.normalize(xcentre_control, 0, self.n_pawns, self.feature_sizes["XCentreControl"]))

        # F10: TotalMobility
        total_mobility = len(self.env.available_moves(state, player))
        features.append(self.normalize(total_mobility, 0, self.n_pawns, self.feature_sizes["TotalMobility"]))

        # F11: Exposure
        exposure = self.exposed_pawns(state, player)
        features.append(self.normalize(exposure, 0, self.n_pawns, self.feature_sizes["Exposure"]))

        # F5: Advancement
        advancement = self.pawn_advancement(state, player)
        features.append(self.normalize(advancement, 0, self.n_pawns, self.feature_sizes["Advancement"]))

        # F6: DoubleDiagonal
        double_diagonal = self.double_diagonal(state, player)
        features.append(self.normalize(double_diagonal, 0, self.n_pawns, self.feature_sizes["DoubleDiagonal"]))

        # F12: KingCentreControl
        king_centre_control = self.center_pawns(state, player, king=True)
        features.append(self.normalize(king_centre_control, 0, self.n_pawns, self.feature_sizes["KingCentreControl"]))
        
        # F13: DiagonalMoment
        diagonal_moment = self.diagonal_movement(state, player)
        features.append(self.normalize(diagonal_moment, 0, self.n_pawns, self.feature_sizes["DiagonalMoment"]))

        # F14: Threat
        opp_threat = self.threatened_pawns(state, self.opponent(player))
        features.append(self.normalize(opp_threat, 0, self.n_pawns, self.feature_sizes["Threat"]))

        # F15: Taken
        taken = self.n_pawns - n_player
        features.append(self.normalize(taken, 0, self.n_pawns, self.feature_sizes["Taken"]))
        
        return np.concatenate(features)

    def normalize(self, value, min_val, max_val, num_bits):
        """
        """
        value = max(min(value, max_val), min_val)
        scaled_value = int(((value - min_val)/(max_val - min_val)) *((2**num_bits-1)))
        return np.array([int(x) for x in format(scaled_value, f'0{num_bits}b')])
    
    def pawns_for(self, player):
        """
        """
        return [CheckersRL.WHITE_PAWN, CheckersRL.WHITE_KING] if player == CheckersRL.WHITE_PAWN else [CheckersRL.BLACK_PAWN, CheckersRL.BLACK_KING]

    def opponent(self, player): 
        """
        """
        return CheckersRL.WHITE_PAWN if player == CheckersRL.BLACK_PAWN else CheckersRL.BLACK_PAWN
    
    def threatened_pawns(self, state, player):
        """
        """
        opp = self.opponent(player)
        threatened = 0
        for move in self.env.available_moves(state, opp):
            if abs(move[0][0] - move[1][0]) == 2:
                threatened += 1
        return threatened
    
    def capture_moves(self, state, player):
        """
        """
        return sum(1 for move in self.env.available_moves(state, player) if abs(move[0][0] - move[1][0]) == 2)
    
    def center_pawns(self, state, player, king=False):
        """
        """
        board_size = self.env.BOARD_SIZE
        assert board_size%2 == 0
        cen1, cen2 = board_size//2 - 1, board_size//2
        center_positons = [(cen1, cen1), (cen1, cen2), (cen2, cen1), (cen2, cen2)]

        count = 0 
        for (row, col) in center_positons:
            tile = state[row][col]
            if king:
                if player == self.env.WHITE_PAWN and tile == self.env.WHITE_KING:
                    count += 1
                elif player == self.env.BLACK_PAWN and tile == self.env.BLACK_KING:
                    count += 1
            else:
                if tile in self.pawns_for(player):
                    count += 1
        return count 
    
    def xcenter_pawns(self, state, player):
        """
        """
        board_size = self.env.BOARD_SIZE
        assert board_size%2 == 0
        cen1, cen2 = board_size//2 - 1, board_size//2
        xcenter_positions = []
        for (cx, cy) in [(cen1, cen1), (cen1, cen2), (cen2, cen1), (cen2, cen2)]:
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (-1, -1)]:
                x, y = cx+dx, cy+dy
                xcenter_positions.append((x, y))
        return sum(1 for (row, col) in xcenter_positions if state[row][col] in self.pawns_for(player))
    
    def exposed_pawns(self, state, player):
        """
        """
        exposed = 0
        board_size = self.env.BOARD_SIZE
        for row in range(board_size):
            for col in range(board_size):
                if state[row][col] in self.pawns_for(player):
                    neighbors = [(row-1, col-1), (row-1, col+1), (row+1, col-1), (row+1, col+1)]
                    if not any((0 <= r < board_size and 0 <= c < board_size and state[r][c] in self.pawns_for(player)) 
                                for r, c in neighbors):
                        exposed += 1
        return exposed
    
    def pawn_advancement(self, state, player):
        advancements = []
        board_size = self.env.BOARD_SIZE
        for row in range(board_size):
            for col in range(board_size):
                if state[row][col] == player:
                    advancements.append(board_size - 1 - row if player == self.env.WHITE_PAWN else row)
        return np.mean(advancements) if advancements else 0

        
    def double_diagonal(self, state, player):
        count = 0
        board_size = self.env.BOARD_SIZE
        for row in range(board_size - 1):
            for col in range(1, board_size - 1):
                if state[row][col] in self.pawns_for(player):
                    if state[row + 1][col + 1] in self.pawns_for(player):
                        count += 1
                    if state[row + 1][col - 1] in self.pawns_for(player):
                        count += 1
        return count

    
    def diagonal_movement(self, state, player):
        """
        """
        return sum(1 for move in self.env.available_moves(state, player) 
                   if abs(move[0][1] - move[1][1]) == 1)
    
    @property
    def size(self):
        """
        """
        return self.feature_size

###############################################################################
# TD(λ) Linear Value function Approximation
###############################################################################
    
class TDLambda_LVFA:
    """
    """
    def __init__(self, env, feature_encoder_cls=CheckersRLFeaturesEncoder, alpha=0.01, alpha_decay=1,
                 gamma=0.9999, epsilon=0.3, epsilon_decay=0.995, final_epsilon=0.2, lambda_=0.9):
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

    def V(self, state):
        """
        """
        feats = self.feature_encoder.encode(state, self.agent_mark)
        return np.dot(self.weights, feats)
    
    def simulate_action(self, state, action, player):
        """
        """
        new_state = deepcopy(state)
        new_state, new_player = self.env.transition_function(new_state, action, player, simulation=True)
        return new_state, new_player
    
    def update_transition(self, s, action, s_prime, reward, done, current_player, next_player):
        """
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
        """
        self.epsilon = max(self.final_epsilon, self.epsilon*self.epsilon_decay)
        self.alpha *= self.alpha_decay

    def policy(self, state, current_player):
        """
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
    
    def epsilon_greedy(self, state, current_player, epsilon=None):
        """
        """
        if epsilon is None:
            epsilon = self.epsilon
        actions = self.env.available_moves(state, current_player)
        if current_player != self.agent_mark:
            return random.choice(actions)
        if random.random() < epsilon:
            return random.choice(actions)
        return self.policy(state, current_player)


    def train(self, episodes=200):
        """
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

    def evaluate(self, episodes=10):
        """
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

    def save(self, filename):
        """
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """
        """
        return pickle.load(open(filename,'rb'))