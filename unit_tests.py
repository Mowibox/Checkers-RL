"""
    @file        unit_tests.py
    @author      Mowibox (Ousmane THIONGANE)
    @brief       Unit tests for CheckersRL environment
    @version     1.0
    @date        2026-05-03
    
"""
# Imports 
import pytest
from copy import deepcopy
from CheckersRL import CheckersRL

def empty_board(env):
    """
    Utility function to create an empty board for testing
    
    @param env: The provided CheckersRL environment
    """
    return [[env.EMPTY_TILE for _ in range(env.BOARD_SIZE)] for _ in range(env.BOARD_SIZE)]

def test_initial_setup():
    """Test the initial board setup and player turn"""
    env = CheckersRL()
    state, player = env.reset()

    white = sum(cell in (env.WHITE_PAWN, env.WHITE_KING) for row in state for cell in row)
    black = sum(cell in (env.BLACK_PAWN, env.BLACK_KING) for row in state for cell in row)

    assert white == 20
    assert black == 20
    assert player == env.WHITE_PAWN

def test_pawn_simple_move():
    """Test that a pawn can move diagonally forward to an empty square"""
    env = CheckersRL()
    state = empty_board(env)

    state[5][2] = env.WHITE_PAWN
    env.current_state = state

    moves = env.available_moves(state, env.WHITE_PAWN)

    assert ((5, 2), (4, 1)) in moves or ((5, 2), (4, 3)) in moves


def test_pawn_capture_forced():
    """
    Test that a pawn must capture if an opponent's piece 
    is adjacent and the landing square is empty
    """
    env = CheckersRL()
    state = empty_board(env)

    state[5][2] = env.WHITE_PAWN
    state[4][3] = env.BLACK_PAWN

    env.current_state = state

    moves = env.available_moves(state, env.WHITE_PAWN)

    assert moves == [((5, 2), (3, 4))]

def test_pawn_multi_capture():
    """Test that a pawn can perform multiple captures in a single turn"""
    env = CheckersRL()
    state = empty_board(env)

    state[5][2] = env.WHITE_PAWN
    state[4][3] = env.BLACK_PAWN
    state[2][5] = env.BLACK_PAWN

    env.current_state = state

    moves = env.available_moves(state, env.WHITE_PAWN)

    assert ((5, 2), (3, 4), (1, 6)) in moves


def test_king_moves():
    """Test that a king can move diagonally in all directions and slide multiple squares"""
    env = CheckersRL()
    state = empty_board(env)

    state[5][2] = env.WHITE_KING
    env.current_state = state

    moves = env.available_moves(state, env.WHITE_PAWN)

    assert ((5, 2), (4, 1)) in moves
    assert ((5, 2), (3, 0)) in moves


def test_king_capture_long_range():
    """
    Test that a king can capture an opponent's piece 
    from a distance and land anywhere beyond
    """
    env = CheckersRL()
    state = empty_board(env)

    state[6][1] = env.WHITE_KING
    state[4][3] = env.BLACK_PAWN

    env.current_state = state

    moves = env.available_moves(state, env.WHITE_PAWN)

    assert any(move[0] == (6, 1) and len(move) == 2 for move in moves)


def test_king_multi_capture_chain():
    """Test that a king can perform multiple captures in a single turn with long-range jumps"""
    env = CheckersRL()
    state = empty_board(env)

    state[7][0] = env.WHITE_KING
    state[5][2] = env.BLACK_PAWN
    state[3][4] = env.BLACK_PAWN

    env.current_state = state

    moves = env.available_moves(state, env.WHITE_PAWN)

    # Must capture both
    assert any(len(move) > 2 for move in moves)


def test_max_capture_rule():
    """
    Test that if multiple capture sequences are available, 
    the one with the most captures must be taken
    """
    env = CheckersRL()
    state = empty_board(env)

    state[7][0] = env.WHITE_KING
    state[5][2] = env.BLACK_PAWN
    state[3][4] = env.BLACK_PAWN
    state[5][6] = env.BLACK_PAWN

    env.current_state = state

    moves = env.available_moves(state, env.WHITE_PAWN)

    # Only longest capture allowed
    lengths = [len(m) for m in moves]
    assert all(l == max(lengths) for l in lengths)

def test_pawn_promotion():
    """Test that a pawn is promoted to a king when it reaches the opponent's back row"""
    env = CheckersRL()
    state = empty_board(env)

    state[1][2] = env.WHITE_PAWN
    env.current_state = state

    action = ((1, 2), (0, 3))
    new_state, _ = env.transition_function(state, action, env.WHITE_PAWN)

    assert new_state[0][3] == env.WHITE_KING


def test_game_end_no_pieces():
    """Test that the game ends when a player has no pieces left"""
    env = CheckersRL()
    state = empty_board(env)

    state[0][1] = env.WHITE_PAWN

    env.current_state = state

    done, winner = env.check_termination(state)

    assert done
    assert winner == env.WHITE_PAWN


def test_draw_by_no_capture():
    """Test that the game ends in a draw if no captures occur for a certain number of turns"""
    env = CheckersRL()

    env.non_capture_action = env.stalemate_threshold

    done, winner = env.check_termination(env.current_state)

    assert done
    assert winner is None