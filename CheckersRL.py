"""
    @file        CheckersRL.py
    @author      Mowibox (Ousmane THIONGANE)
    @brief       Class for the Checkers game
    @version     1.0
    @date        2025-01-20
    
"""
# Imports
import random
import pygame
from copy import deepcopy


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

    def __init__(self, human_play: int=None):
        """
        Initialize a Checkers board

        @param human_play: Enables human playing and specifies which player is controlled
        """
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Checkers")
        self.players = (self.WHITE_PAWN, self.BLACK_PAWN)
        self.human_player = human_play
        self.selected_pawn = None
        self.highlighted_actions = []
        self.reset()


    def switch_player(self, player: int) -> int:
        """
        Gives the opponent of the currrent player

        @param player: The current player
        """
        return self.WHITE_PAWN if player is self.BLACK_PAWN else self.BLACK_PAWN
    

    def check_termination(self, state: list) -> tuple[bool, int]:
        """
        Checks if the games is terminated

        @param state: The current state
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
    

    def available_moves(self) -> list[tuple]:
        """
        Available moves at current state
        """
        actions = []
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                pawn = self.current_state[row][col]
                if pawn == self.current_player or pawn == self.current_player + 1:
                    possible_moves = self.get_available_moves(self.current_state, row, col)
                    for new_row, new_col in possible_moves:
                        actions.append(((row, col), (new_row, new_col)))
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
    

    def transition_function(self, state: list, action: list[tuple], player: int) -> tuple[list, int]:
        """
        Applies a move for a pawn to the board

        @param state: The current state
        @param action: The chosen action
        @param player: The current player
        """
        # Normal move
        (row, col), (new_row, new_col) = action
        pawn = state[row][col]
        state[new_row][new_col] = pawn
        state[row][col] = self.EMPTY_TILE

        # Capture move
        if abs(new_row - row) == 2:
            mid_row, mid_col = (row + new_row)//2, (col + new_col)//2
            state[mid_row][mid_col] = self.EMPTY_TILE
        
        # King promotion
        if (pawn == self.WHITE_PAWN and new_row == 0) or (pawn == self.BLACK_PAWN and new_row == self.BOARD_SIZE-1):
            state[new_row][new_col] = self.WHITE_KING if pawn == self.WHITE_PAWN else self.BLACK_KING

        player = self.switch_player(player)  
        return state, player

    def step(self, action: list[tuple]):
        """
        Step funciton

        @param action: The chosen action
        """

        self.done, winner = self.check_termination(self.current_state)
        assert self.done == False

        self.current_state, self.current_player = self.transition_function(self.current_state, action, self.current_player)
        self.done, winner = self.check_termination(self.current_state)
        reward = 0
        if winner == self.WHITE_PAWN:
            reward = 1
        elif winner == self.BLACK_PAWN:  
            reward = -1
        return self.current_state, reward, self.done, self.current_player

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
                row, col = y//self.TILE_SIZE, x//self.TILE_SIZE

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

        if player == self.BLACK_PAWN:
            self.step(random.choice(self.available_moves))

        return self.current_state, self.current_player