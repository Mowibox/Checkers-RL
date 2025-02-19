"""
    @file        CheckersRL.py
    @author      Mowibox (Ousmane THIONGANE)
    @brief       Class for the Checkers game
    @version     1.0
    @date        2025-01-20
    
"""
# Imports
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = ''
import pygame
import random
import pygame
from copy import deepcopy
from CheckersRLFeaturesEncoder import CheckersRLFeaturesEncoder


class CheckersRL:
    """
    Checkers game class (Draughts)
    """
    EMPTY_TILE = 0
    WHITE_PAWN, WHITE_KING = 1, 2
    BLACK_PAWN, BLACK_KING = 3, 4

    BOARD_SIZE = 8
    TILE_SIZE = 60
    WIDTH, HEIGHT = BOARD_SIZE*TILE_SIZE, BOARD_SIZE*TILE_SIZE

    COLOR = {
        "Light": (255, 205, 160),
        "Dark": (210, 140, 70),
        "White": (255, 255, 255),
        "Black": (0, 0, 0),
        "Yellow": (255, 215, 0),
        "Green": (0, 255, 0),
    }

    def __init__(self, human_play: int=None, stalemate_threshold: int=25):
        """
        Initializes a Checkers board

        @param human_play: Enables human playing and specifies which player is controlled
        @param stalemate_threshold: Set the maximum number of uncaptured actions before the match is drawn
        """
        self.screen = None
        self.players = (self.WHITE_PAWN, self.BLACK_PAWN)
        self.human_player = human_play
        self.selected_pawn = None
        self.highlighted_actions = []
        self.stalemate_threshold = stalemate_threshold
        self.reset()
        self.encoder = CheckersRLFeaturesEncoder(self)


    def switch_player(self, player: int) -> int:
        """
        Returns the opponent of the currrent player

        @param player: The current player
        """
        return self.WHITE_PAWN if player is self.BLACK_PAWN else self.BLACK_PAWN
    

    def check_termination(self, state: list, simulation: bool=False) -> tuple[bool, int]:
        """
        Checks if the games is terminated

        @param state: The current state
        @param simulation: If True, the stalemate_threshold is ignored (e.g. for MCTS simulations)
        """
        done = False
        white_pawns = any(pawn in (self.WHITE_PAWN, self.WHITE_KING) for row in state for pawn in row)
        black_pawns = any(pawn in (self.BLACK_PAWN, self.BLACK_KING) for row in state for pawn in row)

        if not white_pawns:
            done = True
            return done, self.BLACK_PAWN
        if not black_pawns:
            done = True
            return done, self.WHITE_PAWN

        if not simulation and self.non_capture_action >= self.stalemate_threshold:
            done = True
            return done, None
        
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if state[row][col] == self.current_player or state[row][col] == self.current_player + 1:
                    if self.get_available_moves(state, row, col):
                        done = False
                        return done, None 
         
        done = True
        return done, self.switch_player(self.current_player)  
    

    def get_available_moves(self, state: list, row: int, col: int) -> list[tuple]:
        """
        Returns all possible moves for the selected pawn

        @param state: The current state
        @param row: The provided board row
        @param col: The provided board column
        """

        actions = []
        pawn = state[row][col]
        directions = [(1, -1), (1, 1)] if pawn == self.BLACK_PAWN else [(-1, -1), (-1, 1)]

        if pawn in (self.WHITE_KING, self.BLACK_KING):
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_move(state, new_row, new_col):
                actions.append((new_row, new_col))

            capture_row, capture_col = row + 2*dr, col + 2*dc
            if self.is_valid_capture(state, row, col, new_row, new_col, capture_row, capture_col):
                actions.append((capture_row, capture_col))
        
        return actions
    

    def available_moves(self, state: list, player: int) -> list[tuple]:
        """
        Available moves at current state
        """
        if state is None:
            state = self.current_state
        if player is None:
            player = self.current_player

        moves = []
        capture_moves = []
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                pawn = state[row][col]
                if pawn == player or pawn == player + 1:
                    possible_moves = self.get_available_moves(state, row, col)
                    for new_row, new_col in possible_moves:
                        if abs(new_row - row) == 2:
                            capture_moves.append(((row, col), (new_row, new_col)))
                        else:
                            moves.append(((row, col), (new_row, new_col)))
        
        if capture_moves:
            actions = capture_moves
        else:
            actions = moves
        return actions
    

    def is_valid_move(self, state: list, next_row: int, next_col: int) -> bool:
        """
        Checks if a move is valid

        @param state: The current state
        @param next_row: The next row
        @param next_col: The next column
        """
        if not (0 <= next_row < self.BOARD_SIZE and 0 <= next_col < self.BOARD_SIZE):
            return False
        return state[next_row][next_col] == self.EMPTY_TILE
    

    def is_valid_capture(self, state: list, row: int, col: int, mid_row: int, mid_col: int, next_row: int, next_col: int) -> bool:
        """
        Checks if a pawn capture is valid

        @param state: The current state
        @param row: The board row
        @param col: The board col
        @param mid_row: The intermediate row
        @param mid_col: The intermediate col
        @param next_row: The next row
        @param next_col: The next column
        """
        if not (0 <= next_row < self.BOARD_SIZE and 0 <= next_col < self.BOARD_SIZE):
            return False
        opponent_pawns = {self.BLACK_PAWN, self.BLACK_KING} if state[row][col] in (self.WHITE_PAWN, self.WHITE_KING) else {self.WHITE_PAWN, self.WHITE_KING}
        return (state[mid_row][mid_col] in opponent_pawns) and (state[next_row][next_col] == self.EMPTY_TILE)
    

    def transition_function(self, state: list, action: list[tuple], player: int, simulation: bool=False) -> tuple[list, int]:
        """
        Applies a move for a pawn to the board

        @param state: The current state
        @param action: The chosen action
        @param player: The current player
        @param simulation: If True, the stalemate_threshold is ignored (e.g. for MCTS simulations)
        """
        # Normal move
        (row, col), (new_row, new_col) = action
        pawn = state[row][col]
        state[new_row][new_col] = pawn
        state[row][col] = self.EMPTY_TILE

        # Capture move
        capture_occured = False
        if abs(new_row - row) == 2:
            mid_row, mid_col = (row + new_row)//2, (col + new_col)//2
            state[mid_row][mid_col] = self.EMPTY_TILE
            capture_occured = True
        
        # King promotion
        if (pawn == self.WHITE_PAWN and new_row == 0) or (pawn == self.BLACK_PAWN and new_row == self.BOARD_SIZE-1):
            state[new_row][new_col] = self.WHITE_KING if pawn == self.WHITE_PAWN else self.BLACK_KING

        player = self.switch_player(player)

        if not simulation:
            if not capture_occured:
                self.non_capture_action += 1
            else:
                self.non_capture_action = 0

        return state, player
    

    def step(self, action: list[tuple]):
        """
        Step funciton

        @param action: The chosen action
        """
        self.done, winner = self.check_termination(self.current_state)
        assert self.done == False

        prev_state = deepcopy(self.current_state)

        self.current_state, self.current_player = self.transition_function(self.current_state, action, self.current_player)
        self.done, winner = self.check_termination(self.current_state)
        reward = 0
        if winner == self.player:
            reward = 250
        elif winner == self.switch_player(self.player):  
            reward = -250

        reward += self.compute_intermediate_reward(prev_state, self.current_state, action)
        return self.current_state, reward, self.done, self.current_player

    def compute_intermediate_reward(self, prev_state: list, new_state: list, action: list[tuple]) -> float:
        """
        Computes an intermediate reward based on the transition between two states

        @param prev_state: The state before the action
        @param new_state: The state after the action
        @param action: The chosen action
        """
        reward = 0.0
        player = self.player
        opponent = self.switch_player(self.player)

        # Reward for pawn advantage
        prev_n_player = sum(1 for row in prev_state for tile in row if tile == self.encoder.pawns_for(player)[0])
        + sum(2 for row in prev_state for tile in row if tile == self.encoder.pawns_for(player)[1])
        prev_n_opp = sum(1 for row in prev_state for tile in row if tile == self.encoder.pawns_for(opponent)[0])
        + sum(2 for row in prev_state for tile in row if tile == self.encoder.pawns_for(opponent)[1])
        new_n_player = sum(1 for row in new_state for tile in row if tile == self.encoder.pawns_for(player)[0])
        + sum(2 for row in new_state for tile in row if tile == self.encoder.pawns_for(player)[1])
        new_n_opp = sum(1 for row in new_state for tile in row if tile == self.encoder.pawns_for(opponent)[0])
        + sum(2 for row in new_state for tile in row if tile == self.encoder.pawns_for(opponent)[1])
        prev_advantage = prev_n_player - prev_n_opp
        new_advantage = new_n_player - new_n_opp
        reward += 0.5*(new_advantage - prev_advantage)

        # Reward for having less pawns threatened by the opponent's pawns
        prev_threat = self.encoder.threatened_pawns(prev_state, player)
        new_threat = self.encoder.threatened_pawns(new_state, player)
        reward += 0.05*(prev_threat - new_threat)

        # Reward for having available capture moves
        prev_capture = self.encoder.capture_moves(prev_state, player)
        new_capture = self.encoder.capture_moves(new_state, player)
        reward += 0.05*(new_capture - prev_capture)

        # Reward for aligned pawns
        prev_dd = self.encoder.double_diagonal(prev_state, player)
        new_dd = self.encoder.double_diagonal(new_state, player)
        reward += 0.1*(new_dd - prev_dd)

        # Reward for having pawns on the last row
        if player == self.WHITE_PAWN:
            prev_back_count = sum(1 for tile in prev_state[-1] if tile == self.WHITE_PAWN)
            new_back_count = sum(1 for tile in new_state[-1] if tile == self.WHITE_PAWN)
        else:
            prev_back_count = sum(1 for tile in prev_state[0] if tile == self.BLACK_PAWN)
            new_back_count = sum(1 for tile in new_state[0] if tile == self.BLACK_PAWN)
            
        reward += 0.005*new_back_count - 0.005*(prev_back_count - new_back_count)

        # Reward for controlling the center
        prev_center = self.encoder.center_pawns(prev_state, player)
        new_center = self.encoder.center_pawns(new_state, player)
        reward += 0.2*(new_center - prev_center)

        # Reward for controlling the center with kings
        prev_kcenter = self.encoder.center_pawns(prev_state, player, king=True)
        new_kcenter = self.encoder.center_pawns(new_state, player, king=True)
        reward += 0.2*(new_kcenter - prev_kcenter)

        return reward
    
    
    def human_input(self) -> tuple[list, int, bool, int]:
        """
        Handles the human player inputs
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.MOUSEBUTTONDOWN and self.current_player == self.human_player:
                x, y = pygame.mouse.get_pos()
                row, col = y // self.TILE_SIZE, x // self.TILE_SIZE
                
                capture_moves_dict = {}
                for r in range(self.BOARD_SIZE):
                    for c in range(self.BOARD_SIZE):
                        if self.current_state[r][c] == self.current_player or self.current_state[r][c] == self.current_player + 1:
                            possible_moves = self.get_available_moves(self.current_state, r, c)
                            capture_moves = [(new_r, new_c) for (new_r, new_c) in possible_moves if abs(new_r - r) == 2]
                            if capture_moves:
                                capture_moves_dict[(r,c)] = capture_moves

                if capture_moves_dict:
                    if self.selected_pawn:
                        if self.selected_pawn in capture_moves_dict:
                            self.highlighted_actions = capture_moves_dict[self.selected_pawn]
                            if (row, col) in self.highlighted_actions:
                                action = (self.selected_pawn, (row, col))
                                state, done, reward, player = self.step(action)
                                self.done = done
                                self.selected_pawn = None
                                self.highlighted_actions = []
                                return state, done, reward, player
                            else:
                                self.selected_pawn = None
                                self.highlighted_actions = []
                        else:
                            self.selected_pawn = None
                            self.highlighted_actions = []
                    else:
                        if (row, col) in capture_moves_dict:
                            self.selected_pawn = (row, col)
                            self.highlighted_actions = capture_moves_dict[(row, col)]
                        else:
                            self.selected_pawn = None
                            self.highlighted_actions = []

                else:
                    if self.selected_pawn:
                        if (row, col) in self.highlighted_actions:
                            action = (self.selected_pawn, (row, col))
                            state, done, reward, player = self.step(action)
                            self.done = done
                            self.selected_pawn = None
                            self.highlighted_actions = []
                            return state, done, reward, player
                        else:
                            self.selected_pawn = None
                            self.highlighted_actions = []
                    else:
                        if self.current_state[row][col] in (self.human_player, self.human_player + 1):
                            self.selected_pawn = (row, col)
                            self.highlighted_actions = self.get_available_moves(self.current_state, row, col)

        return None, None, self.done, None


    def render(self, state: list=None):
        """
        Renders the game in a pygame window

        @param state: The current state
        """
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Checkers")

        self.screen.fill((0, 0, 0))

        if state is None:
            state = self.current_state
        
        # Board
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                color = self.COLOR["Dark"] if (row + col)%2 != 0 else self.COLOR["Light"]
                pygame.draw.rect(self.screen, color,
                                 (col*self.TILE_SIZE, row*self.TILE_SIZE,
                                 self.TILE_SIZE, self.TILE_SIZE))
                
        # Pawn
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                pawn = state[row][col]
                if pawn != self.EMPTY_TILE:
                    pawn_color = self.COLOR["White"] if pawn in (self.WHITE_PAWN, self.WHITE_KING) else self.COLOR["Black"]
                    pygame.draw.circle(self.screen, pawn_color,
                                       (col*self.TILE_SIZE + self.TILE_SIZE//2,
                                        row*self.TILE_SIZE + self.TILE_SIZE//2),
                                        self.TILE_SIZE//3)
                    if pawn in (self.WHITE_KING, self.BLACK_KING):
                        pygame.draw.circle(self.screen, self.COLOR["Yellow"],
                                            (col*self.TILE_SIZE + self.TILE_SIZE//2,
                                            row*self.TILE_SIZE + self.TILE_SIZE//2),
                                            self.TILE_SIZE//4, 3)
        # Highlighted actions
        if self.highlighted_actions:
            for row, col in self.highlighted_actions:
                pygame.draw.rect(self.screen, self.COLOR["Green"],
                                 (col*self.TILE_SIZE, row*self.TILE_SIZE,
                                  self.TILE_SIZE, self.TILE_SIZE), 3)
        pygame.display.flip()
        

    def reset(self, player: int=None) -> tuple[list, int]:
        """
        Resets the board to the initial state

        @param player: The current player
        """

        self.board = [[self.EMPTY_TILE for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        for row in range(0, self.BOARD_SIZE//2 -1):
            for col in range(row%2, self.BOARD_SIZE+1, 2):
                if col == 0:
                    continue
                self.board[row][col-1] = self.BLACK_PAWN

        for row in range(self.BOARD_SIZE//2+1, self.BOARD_SIZE):
            for col in range(row%2, self.BOARD_SIZE+1, 2):
                if col == 0:
                    continue
                self.board[row][col-1] = self.WHITE_PAWN

        self.done = False
        self.player = player
        if self.player is None:
            self.player = self.WHITE_PAWN

        self.current_state = deepcopy(self.board)
        self.current_player = self.WHITE_PAWN
        self.non_capture_action = 0

        if player == self.BLACK_PAWN:
            self.step(random.choice(self.available_moves(self.current_state, self.current_player)))

        return self.current_state, self.current_player