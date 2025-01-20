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
    WHITE_PIECE, WHITE_KING = 1, 2
    BLACK_PIECE, BLACK_KING = 3, 4

    BOARD_SIZE = 8
    TILE_SIZE = 60
    WIDTH, HEIGHT = BOARD_SIZE*TILE_SIZE, BOARD_SIZE*TILE_SIZE

    COLOR = {
        "Light": (255, 205, 160),
        "Dark": (210, 140, 70),
        "White": (255, 255, 255),
        "Black": (0, 0, 0),
        "Yellow": (255, 215, 0),
    }

    def __init__(self):
        """
        Initialize a Checkers board
        """
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Checkers")
        self.players = (self.WHITE_PIECE, self.BLACK_PIECE)
        self.reset()


    def switch_player(self, player: int) -> int:
        """
        Gives the opponent of the currrent player

        @param player: The current player
        """
        return self.WHITE_PIECE if player is self.BLACK_PIECE else self.BLACK_PIECE
    

    def check_termination(self, state: list) -> tuple[bool, int]:
        """
        Checks if the games is terminated

        @param state: The current state
        """
        done = False
        white_pieces = any(piece in (self.WHITE_PIECE, self.WHITE_KING) for row in state for piece in row)
        black_pieces = any(piece in (self.BLACK_PIECE, self.BLACK_KING) for row in state for piece in row)

        if not white_pieces:
            done = True
            return done, self.BLACK_PIECE
        if not black_pieces:
            done = True
            return done, self.WHITE_PIECE
        
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
        Returns all possible moves for the selected piece

        @param state: The current state
        @param row: The provided board row
        @param col: The provided board column
        """

        actions = []
        piece = state[row][col]
        directions = [(1, -1), (1, 1)] if piece == self.BLACK_PIECE else [(-1, -1), (-1, 1)]

        if piece in (self.WHITE_KING, self.BLACK_KING):
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
                piece = self.current_state[row][col]
                if piece == self.current_player or piece == self.current_player + 1:
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
        Checks if a piece capture is valid

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
        opponent_pieces = {self.BLACK_PIECE, self.BLACK_KING} if state[row][col] in (self.WHITE_PIECE, self.WHITE_KING) else {self.WHITE_PIECE, self.WHITE_KING}
        return (state[mid_row][mid_col] in opponent_pieces) and (state[next_row][next_col] == self.EMPTY_TILE)
    

    def transition_function(self, state: list, action: list[tuple], player: int) -> tuple[list, int]:
        """
        Applies a move for a piece to the board

        @param state: The current state
        @param action: The chosen action
        @param player: The current player
        """
        # Normal move
        (row, col), (new_row, new_col) = action
        piece = state[row][col]
        state[new_row][new_col] = piece
        state[row][col] = self.EMPTY_TILE

        # Capture move
        if abs(new_row - row) == 2:
            mid_row, mid_col = (row + new_row)//2, (col + new_col)//2
            state[mid_row][mid_col] = self.EMPTY_TILE
        
        # King promotion
        if (piece == self.WHITE_PIECE and new_row == 0) or (piece == self.BLACK_PIECE and new_row == self.BOARD_SIZE-1):
            state[new_row][new_col] = self.WHITE_KING if piece == self.WHITE_PIECE else self.BLACK_KING

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
        if winner == self.WHITE_PIECE:
            reward = 1
        elif winner == self.BLACK_PIECE:  
            reward = -1
        return self.current_state, reward, self.done, self.current_player
        

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
                
        # Pieces
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                piece = state[row][col]
                if piece != self.EMPTY_TILE:
                    piece_color = self.COLOR["White"] if piece in (self.WHITE_PIECE, self.WHITE_KING) else self.COLOR["Black"]
                    pygame.draw.circle(self.screen, piece_color,
                                       (col*self.TILE_SIZE + self.TILE_SIZE//2,
                                        row*self.TILE_SIZE + self.TILE_SIZE//2),
                                        self.TILE_SIZE//3)
                    if piece in (self.WHITE_KING, self.BLACK_KING):
                        pygame.draw.circle(self.screen, self.COLOR["Yellow"],
                                            (col*self.TILE_SIZE + self.TILE_SIZE//2,
                                            row*self.TILE_SIZE + self.TILE_SIZE//2),
                                            self.TILE_SIZE//4, 3)
        pygame.display.flip()


    def reset(self, player=None):
        """
        Resets the board to the initial state
        """

        self.board = [[self.EMPTY_TILE for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        for row in range(0, self.BOARD_SIZE//2 -1):
            for col in range(row%2, self.BOARD_SIZE+1, 2):
                if col == 0:
                    continue
                self.board[row][col-1] = self.BLACK_PIECE

        for row in range(self.BOARD_SIZE//2+1, self.BOARD_SIZE):
            for col in range(row%2, self.BOARD_SIZE+1, 2):
                if col == 0:
                    continue
                self.board[row][col-1] = self.WHITE_PIECE

        self.done = False
        self.player = player
        if self.player is None:
            self.player = self.WHITE_PIECE

        self.current_state = deepcopy(self.board)
        self.current_player = self.WHITE_PIECE

        if player == self.BLACK_PIECE:
            self.step(random.choice(self.available_moves))

        return self.current_state, self.current_player
    
