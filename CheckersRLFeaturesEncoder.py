"""
@file        CheckersRLFeaturesEncoder.py
@author      Mowibox (Ousmane THIONGANE)
@brief       Features Encoder for the self.env.CheckersRL environment
@version     1.0
@date        2025-02-18
"""
# Imports
import numpy as np

class CheckersRLFeaturesEncoder:
    """
    Features Encoder for the self.env.CheckersRL class
    """
    def __init__(self, env):
        """
        Initializes the feature encoder

        @param env: the self.env.CheckersRL environment
        """
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
            "Taken": 3,
        }
        self.feature_size = sum(self.feature_sizes.values())
        self.n_pawns = sum(tile == self.env.WHITE_PAWN for row in self.env.board for tile in row)

    
    def encode(self, state: list, player: int) -> np.ndarray:
        """
        Encodes a given game state into a feature vector

        @param state: The provided state 
        @param player: The current player
        """
        features = []

        player_pawns = self.pawns_for(player)
        opponent_pawns = self.pawns_for(self.opponent(player))

        # == F1: PawnAdvantage ==
        n_player = sum(1 for row in state for tile in row if tile in player_pawns)
        n_opponent = sum(1 for row in state for tile in row if tile in opponent_pawns)
        pawn_advantage = n_player - n_opponent
        features.append(self.normalize(pawn_advantage, -self.n_pawns, self.n_pawns, self.feature_sizes["PawnAdvantage"]))

        # == F2: PawnDisadvantage ==
        features.append(self.normalize(-pawn_advantage, -self.n_pawns, self.n_pawns, self.feature_sizes["PawnDisadvantage"]))

        # == F3: PawnThreat ==
        pawn_threat = self.threatened_pawns(state, player)
        features.append(self.normalize(pawn_threat, 0, self.n_pawns, self.feature_sizes["PawnThreat"]))

        # == F4: PawnTake ==
        pawn_take = self.capture_moves(state, player)
        features.append(self.normalize(pawn_take, 0, self.n_pawns, self.feature_sizes["PawnTake"]))

        # == F7: Backrowbridge ==
        if player == self.env.WHITE_PAWN:
            backrow = any(tile == self.env.WHITE_PAWN for tile in state[-1])
        else: 
            backrow = any(tile == self.env.BLACK_PAWN for tile in state[0])
        features.append(np.array([1])) if backrow else features.append(np.array([0]))

        # == F8: Centrecontrol ==
        centre_control = self.center_pawns(state, player)
        features.append(self.normalize(centre_control, 0, self.n_pawns, self.feature_sizes["CentreControl"]))

        # == F9: XCentrecontrol ==
        xcentre_control = self.xcenter_pawns(state, player)
        features.append(self.normalize(xcentre_control, 0, self.n_pawns, self.feature_sizes["XCentreControl"]))

        # == F10: TotalMobility ==
        total_mobility = len(self.env.available_moves(state, player))
        features.append(self.normalize(total_mobility, 0, self.n_pawns, self.feature_sizes["TotalMobility"]))

        # == F11: Exposure ==
        exposure = self.exposed_pawns(state, player)
        features.append(self.normalize(exposure, 0, self.n_pawns, self.feature_sizes["Exposure"]))

        # == F5: Advancement ==
        advancement = self.pawn_advancement(state, player)
        features.append(self.normalize(advancement, 0, self.n_pawns, self.feature_sizes["Advancement"]))

        # == F6: DoubleDiagonal ==
        double_diagonal = self.double_diagonal(state, player)
        features.append(self.normalize(double_diagonal, 0, self.n_pawns, self.feature_sizes["DoubleDiagonal"]))

        # == F12: KingCentreControl ==
        king_centre_control = self.center_pawns(state, player, king=True)
        features.append(self.normalize(king_centre_control, 0, self.n_pawns, self.feature_sizes["KingCentreControl"]))
        
        # == F13: DiagonalMoment ==
        diagonal_moment = self.diagonal_movement(state, player)
        features.append(self.normalize(diagonal_moment, 0, self.n_pawns, self.feature_sizes["DiagonalMoment"]))

        # == F15: Taken ==
        taken = self.n_pawns - n_player
        features.append(self.normalize(taken, 0, self.n_pawns, self.feature_sizes["Taken"]))
        
        return np.concatenate(features)

    def normalize(self, value: int, min_val: int, max_val: int, num_bits: int) -> np.ndarray:
        """
        Normalizes a value to a binary representation

        @param value: The input value
        @param min_val: The minimum possible value
        @param max_val: The maximum possible value
        @param: num_bits: The number of bits for encoding 
        """
        value = max(min(value, max_val), min_val)
        scaled_value = int(((value - min_val)/(max_val - min_val)) *((2**num_bits-1)))
        return np.array([int(x) for x in format(scaled_value, f'0{num_bits}b')])
    
    def pawns_for(self, player: int) -> list[int, int]:
        """
        Returns the pawns of the provided player

        @param player: The current player
        """
        return [self.env.WHITE_PAWN, self.env.WHITE_KING] if player == self.env.WHITE_PAWN else [self.env.BLACK_PAWN, self.env.BLACK_KING]

    def opponent(self, player: int) -> int: 
        """
        Return the opponent of the provided player

        @param player: The current player
        """
        return self.env.WHITE_PAWN if player == self.env.BLACK_PAWN else self.env.BLACK_PAWN
    
    def threatened_pawns(self, state: list, player: int) -> int:
        """
        Returns the number of pawns threatened by an opponent's move

        @param state: The provided state
        @param player: The current player        
        """
        opp = self.opponent(player)
        threatened = 0
        for move in self.env.available_moves(state, opp):
            if abs(move[0][0] - move[1][0]) == 2:
                threatened += 1
        return threatened
    
    def capture_moves(self, state: list, player: int) -> int:
        """
        Returns the number of possible capture moves for the provided player 

        @param state: The provided state
        @param player: The current player
        """
        return sum(1 for move in self.env.available_moves(state, player) if abs(move[0][0] - move[1][0]) == 2)
    
    def center_pawns(self, state: list, player: int, king: bool=False) -> int:
        """
        Returns the number of pawns in the central tiles for the provided player

        @param state: The provided state
        @param player: The current player
        @param king: Counts the number of king on the center if True
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
    
    def xcenter_pawns(self, state: list, player: int) -> int:
        """
        Returns the numbers of a player's pawns positioned adjacent to the central tiles

        @param state: The provided state
        @param player: The current player
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
    
    def exposed_pawns(self, state: list, player: int) -> int:
        """
        Returns the number of pawns that aren't protected by any allied pawns for the provided player

        @param state: The provided state
        @param player: The current player
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
    
    def pawn_advancement(self, state: list, player: int) -> float:
        """
        Returns the average advancement of a player's pawns on the board

        @param state: The provided state
        @param player: The current player
        """
        advancements = []
        board_size = self.env.BOARD_SIZE
        for row in range(board_size):
            for col in range(board_size):
                if state[row][col] == player:
                    advancements.append(board_size - 1 - row if player == self.env.WHITE_PAWN else row)
        return np.mean(advancements) if advancements else 0

        
    def double_diagonal(self, state: list, player: int) -> int:
        """
        Returns the number of times two of a player's pawns are aligned diagonally

        @param state: The provided state
        @param player: The current player
        """
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

    
    def diagonal_movement(self, state: list, player: int) -> int:
        """
        Returns the number of diagonal moves available for the provided player

        @param state: The provided state
        @param player: The current player
        """
        return sum(1 for move in self.env.available_moves(state, player) 
                   if abs(move[0][1] - move[1][1]) == 1)
    
    @property
    def size(self) -> int:
        """
        Returns the size of the feature representation vector
        """
        return self.feature_size