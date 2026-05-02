"""
    @file        CheckersRL.py
    @author      Mowibox (Ousmane THIONGANE)
    @brief       Class for the Checkers game (international variant)
    @version     1.0
    @date        2024-05-03
"""
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = ''
import pygame
import random
from copy import deepcopy
from collections import defaultdict
class CheckersRL:
    """
    Checkers game class (Draughts)
    """
    EMPTY_TILE  = 0
    WHITE_PAWN, WHITE_KING  = 1, 2
    BLACK_PAWN, BLACK_KING  = 3, 4
    BOARD_SIZE  = 10
    TILE_SIZE   = 60
    WIDTH       = BOARD_SIZE * TILE_SIZE
    HEIGHT      = BOARD_SIZE * TILE_SIZE

    DRAW_REPETITION_LIMIT = 3

    COLOR = {
        "Light"  : (255, 205, 160),
        "Dark"   : (210, 140,  70),
        "White"  : (255, 255, 255),
        "Black"  : (0,   0,   0),
        "Green"  : (0,   255, 0),
        "Flare"  : (255, 99,  49),
        "Bolt"   : (0,   198, 255),
        "Selected": (255, 220, 0),
    }

    def __init__(self, human_play: int = None, stalemate_threshold: int = 25) -> None:
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
        self.non_capture_action: int = 0
        self._position_history = defaultdict(int)
        self.reset()

    def reset(self, player: int) -> tuple[list, int]:
        """
        Reset the board to the initial state

        @param player: The current player
        """
        board = [[self.EMPTY_TILE] * self.BOARD_SIZE for _ in range(self.BOARD_SIZE)]

        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if (row + col) % 2 == 1:
                    if row < 4:
                        board[row][col] = self.BLACK_PAWN
                    elif row > 5:
                        board[row][col] = self.WHITE_PAWN

        self.board = board
        self.done = False
        self.player = player if player is not None else self.WHITE_PAWN
        
        self.current_state = deepcopy(board)
        self.current_player = self.WHITE_PAWN
        self.non_captures_action = 0

        self._position_history = defaultdict(int)
        self._record_position(self.current_state, self.current_player)

        if player == self.BLACK_PAWN:
            first_moves = self.available_moves(self.current_state, self.current_player)
            if first_moves:
                action = random.choice(first_moves)
                self.current_state, self.current_player = self.transition_function(
                    self.current_state, action, self.current_player
                )

        return self.current_state, self.current_player

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def switch_player(player: int) -> int:
        """
        Returns the opponent of the current player

        @param player: The current player
        """
        return CheckersRL.WHITE_PAWN if player == CheckersRL.BLACK_PAWN else CheckersRL.BLACK_PAWN

    @staticmethod
    def _is_active(row: int, col: int) -> bool:
        """Checks if the square at (row, col) is playable (dark tile)
        
        @param row: The provided board row
        @param col: The provided board column
        """
        return (row + col) % 2 == 1

    def _is_white(self, pawn: int) -> bool:
        """Checks if the pawn belongs to the white player
        
        @param pawn: The provided pawn
        """
        return pawn in (self.WHITE_PAWN, self.WHITE_KING)

    def _is_black(self, pawn: int) -> bool:
        """Checks if the pawn belongs to the black player
        
        @param pawn: The provided pawn
        """
        return pawn in (self.BLACK_PAWN, self.BLACK_KING)

    def _is_opponent(self, pawn: int, player: int) -> bool:
        """Checks if the pawn belongs to the opponent of the current player

        @param pawn: The provided pawn
        @param player: The current player
        """
        if player == self.WHITE_PAWN:
            return self._is_black(pawn)
        return self._is_white(pawn)

    def _is_own(self, pawn: int, player: int) -> bool:
        """Checks if the pawn belongs to the current player

        @param pawn: The provided pawn
        @param player: The current player
        """
        if player == self.WHITE_PAWN:
            return self._is_white(pawn)
        return self._is_black(pawn)

    def _state_hash(self, state: list, player: int) -> str:
        """Compact string hash of (state, player) for repetition detection

        @param state: The provided board state 
        @param player: The current player
        """
        flat = "".join(str(cell) for row in state for cell in row)
        return flat + str(player)

    def _record_position(self, state: list, player: int) -> None:
        """Record the current position for repetition detection

        @param state: The provided board state
        @param player: The current player
        """
        self._position_history[self._state_hash(state, player)] += 1

    def _is_draw(self) -> bool:
        """Checks if the current position is a draw by repetition or no-capture stalemate"""
        if self.non_capture_action >= self.stalemate_threshold:
            return True
        if any(v >= self.DRAW_REPETITION_LIMIT for v in self._position_history.values()):
            return True
        return False

    def _pawn_directions(self, player: int) -> list[tuple[int, int]]:
        """
        Returns the forward movement directions for pawns of the given player

        @param player: The current player
        """
        if player == self.WHITE_PAWN:
            return [(-1, -1), (-1, 1)]
        return [(1, -1), (1, 1)]

    def _all_directions(self) -> list[tuple[int, int]]:
        """Returns all 4 diagonal directions (for kings)"""
        return [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def _quiet_moves_for_pawn(
        self, state: list, row: int, col: int, player: int
    ) -> list[tuple]:
        """
        Returns quiet (non-capture) actions for the pawn at (row, col)
        Kings slide on the full diagonal (flying king)

        @param state: The provided board state
        @param row: The provided board row 
        @param col: The provided board column
        @param player: The current player
        """
        pawn = state[row][col]
        actions = []

        if pawn in (self.WHITE_KING, self.BLACK_KING):
            for dr, dc in self._all_directions():
                r, c = row + dr, col + dc
                while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE:
                    if state[r][c] == self.EMPTY_TILE:
                        actions.append(((row, col), (r, c)))
                    else:
                        break
                    r += dr
                    c += dc
        else:
            for dr, dc in self._pawn_directions(player):
                r, c = row + dr, col + dc
                if (
                    0 <= r < self.BOARD_SIZE
                    and 0 <= c < self.BOARD_SIZE
                    and state[r][c] == self.EMPTY_TILE
                ):
                    actions.append(((row, col), (r, c)))
        return actions

    def _capture_dfs(
        self,
        state: list,
        row: int,
        col: int,
        player: int,
        captured_set: frozenset[tuple[int, int]],
        path: tuple[tuple[int, int], ...],
    ) -> list[tuple[int, tuple[tuple[int, int], ...]]]:
        """
        Depth-first search for all maximal capture sequences starting at (row, col).

        @param state: The provided board state
        @param row: The provided board row
        @param col: The provided board column
        @param player: The current player
        @param captured_set: The set of already captured pawns in the current path (to prevent double-jumping)
        @param path: The current action path (origin + landing squares)
        """
        pawn = state[row][col]
        is_king = pawn in (self.WHITE_KING, self.BLACK_KING)
        directions = self._all_directions()

        results: list[tuple[int, tuple[tuple[int, int], ...]]] = []
        found_any = False

        for dr, dc in directions:
            if is_king:
                r, c = row + dr, col + dc

                while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE:
                    sq = state[r][c]

                    if sq == self.EMPTY_TILE:
                        r += dr
                        c += dc
                        continue

                    elif self._is_opponent(sq, player) and (r, c) not in captured_set:
                        land_r, land_c = r + dr, c + dc

                        while (
                            0 <= land_r < self.BOARD_SIZE
                            and 0 <= land_c < self.BOARD_SIZE
                            and state[land_r][land_c] == self.EMPTY_TILE
                        ):
                            new_captured = captured_set | {(r, c)}
                            new_path = path + ((land_r, land_c),)

                            sub = self._capture_dfs(
                                state,
                                land_r,
                                land_c,
                                player,
                                frozenset(new_captured),
                                new_path
                            )

                            if sub:
                                found_any = True
                                results.extend(sub)
                            else:
                                found_any = True
                                results.append((len(new_captured), new_path))

                            land_r += dr
                            land_c += dc

                        break

                    elif (r, c) in captured_set:
                        r += dr
                        c += dc
                        continue

                    else:
                        break

            else:
                mid_r, mid_c = row + dr, col + dc
                land_r, land_c = row + 2 * dr, col + 2 * dc

                if not (0 <= land_r < self.BOARD_SIZE and 0 <= land_c < self.BOARD_SIZE):
                    continue

                mid_sq = state[mid_r][mid_c]

                if (
                    self._is_opponent(mid_sq, player)
                    and (mid_r, mid_c) not in captured_set
                    and state[land_r][land_c] == self.EMPTY_TILE
                ):
                    new_captured = captured_set | {(mid_r, mid_c)}
                    new_path = path + ((land_r, land_c),)

                    sub = self._capture_dfs(
                        state,
                        land_r,
                        land_c,
                        player,
                        frozenset(new_captured),
                        new_path
                    )

                    if sub:
                        found_any = True
                        results.extend(sub)
                    else:
                        found_any = True
                        results.append((len(new_captured), new_path))

        return results if found_any else []

    def available_moves(
        self,
        state: list,
        player: int,
    ) -> list[tuple]:
        """
        Available moves at the current state

        @param state: The provided board state
        @param player: The current player
        """
        all_captures: list[tuple[int, tuple]] = []

        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                pawn = state[r][c]
                if not self._is_own(pawn, player):
                    continue
                seqs = self._capture_dfs(
                    state, r, c, player,
                    frozenset(), ((r, c),)
                )
                all_captures.extend(seqs)

        if all_captures:
            max_cap = max(cnt for cnt, _ in all_captures)
            return [path for cnt, path in all_captures if cnt == max_cap]

        quiet = []
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                pawn = state[r][c]
                if self._is_own(pawn, player):
                    quiet.extend(self._quiet_moves_for_pawn(state, r, c, player))

        return quiet

    def get_available_moves(
        self,
        state: list,
        row: int,
        col: int,
    ) -> list[tuple[int, int]]:
        """
        Returns all possible moves for the selected pawn

        @param state: The current state
        @param row: The provided board row
        @param col: The provided board column
        """
        player = (
            self.WHITE_PAWN
            if state[row][col] in (self.WHITE_PAWN, self.WHITE_KING)
            else self.BLACK_PAWN
        )
        all_actions = self.available_moves(state, player)
        destinations: list[tuple[int, int]] = []
        for action in all_actions:
            if action[0] == (row, col):
                destinations.append(action[-1]) 
        return destinations
    
    def transition_function(
        self,
        state: list,
        action: tuple,
        player: int,
        simulation: bool = False,
    ) -> tuple[list, int]:
        """
        Applies a move for a pawn to the board

        @param state: The current state
        @param action: The chosen action
        @param player: The current player
        @param simulation: If True, the stalemate_threshold is ignored (e.g. for MCTS simulations)
        """
        origin = action[0]
        landings = action[1:]

        r0, c0 = origin
        pawn = state[r0][c0]

        captured: list[tuple[int, int]] = []

        cur_r, cur_c = r0, c0  

        for land_r, land_c in landings:
            dr = land_r - cur_r
            dc = land_c - cur_c
            step_r = 1 if dr > 0 else -1
            step_c = 1 if dc > 0 else -1

            r, c = cur_r + step_r, cur_c + step_c
            while (r, c) != (land_r, land_c):
                if state[r][c] != self.EMPTY_TILE:
                    if (r, c) not in captured:
                        captured.append((r, c))
                    break
                r += step_r
                c += step_c

            cur_r, cur_c = land_r, land_c

        # Apply move
        state[r0][c0] = self.EMPTY_TILE
        final_r, final_c = landings[-1]
        state[final_r][final_c] = pawn

        # Remove all captured pawns
        for cr, cc in captured:
            state[cr][cc] = self.EMPTY_TILE

        # King promotion
        back_rank_white = 0
        back_rank_black = self.BOARD_SIZE - 1
        if pawn == self.WHITE_PAWN and final_r == back_rank_white:
            state[final_r][final_c] = self.WHITE_KING
        elif pawn == self.BLACK_PAWN and final_r == back_rank_black:
            state[final_r][final_c] = self.BLACK_KING

        # Draw
        if not simulation:
            if captured:
                self.non_capture_action = 0
            elif pawn in (self.WHITE_KING, self.BLACK_KING):
                self.non_capture_action += 1
            else:
                self.non_capture_action = 0

        next_player = self.switch_player(player)

        if not simulation:
            self._record_position(state, next_player)

        return state, next_player

    def check_termination(
        self,
        state: list,
        simulation: bool = False,
    ) -> tuple[bool, int]:
        """
        Returns (done, winner_or_None).
        winner = WHITE_PAWN | BLACK_PAWN if a player won, None for a draw.

        @param state: The current state
        @param simulation: If True, the stalemate_threshold is ignored (e.g. for MCTS simulations)
        """
        has_white = any(
            state[r][c] in (self.WHITE_PAWN, self.WHITE_KING)
            for r in range(self.BOARD_SIZE) for c in range(self.BOARD_SIZE)
        )
        has_black = any(
            state[r][c] in (self.BLACK_PAWN, self.BLACK_KING)
            for r in range(self.BOARD_SIZE) for c in range(self.BOARD_SIZE)
        )

        if not has_white:
            return True, self.BLACK_PAWN
        if not has_black:
            return True, self.WHITE_PAWN

        # Draw conditions 
        if not simulation and self._is_draw():
            return True, None

        # No legal moves
        if not self.available_moves(state, self.current_player):
            return True, self.switch_player(self.current_player)

        return False, None

    def step(
        self,
        action: tuple,
    ) -> tuple[list, float, bool, int]:
        """
        Step function

        @param action: The chosen action
        """
        assert not self.done, "Game is already over. Call reset()."

        prev_state  = deepcopy(self.current_state)

        self.current_state, self.current_player = self.transition_function(
            self.current_state, action, self.current_player
        )
        self.done, winner = self.check_termination(self.current_state)

        reward = 0.0
        if winner == self.player:
            reward = 250.0
        elif winner == self.switch_player(self.player):
            reward = -250.0
        elif winner is None:
            reward = -50.0

        reward += self.compute_intermediate_reward(prev_state, self.current_state, action)

        return self.current_state, reward, self.done, self.current_player


    def compute_intermediate_reward(
        self,
        prev_state: list,
        new_state: list,
        action: tuple,
    ) -> float:
        """
        Computes an intermediate reward based on the transition between two states

        @param prev_state: The state before the action
        @param new_state: The state after the action
        @param action: The chosen action
        """
        player = self.player
        opponent = self.switch_player(player)

        def count_pawns(state: list, player: int) -> tuple[int, int]:
            pawns = kings = 0
            pawn_id = self.WHITE_PAWN if player == self.WHITE_PAWN else self.BLACK_PAWN
            king_id = self.WHITE_KING if player == self.WHITE_PAWN else self.BLACK_KING
            for row in state:
                for cell in row:
                    if cell == pawn_id:   pawns += 1
                    elif cell == king_id: kings += 1
            return pawns, kings

        prev_pp, prev_pk = count_pawns(prev_state, player)
        prev_op, prev_ok = count_pawns(prev_state, opponent)
        new_pp,  new_pk  = count_pawns(new_state,  player)
        new_op,  new_ok  = count_pawns(new_state,  opponent)

        prev_score = prev_pp + 2 * prev_pk - prev_op - 2 * prev_ok
        new_score  = new_pp  + 2 * new_pk  - new_op  - 2 * new_ok
        reward = 7.5 * (new_score - prev_score)

        reward += 5.0 * (new_pk - prev_pk)

        return reward


    def render(self, state: list = None) -> None:
        """
        Renders the game in a pygame window

        @param state: The provided board state
        """
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("International Checkers")

        if state is None:
            state = self.current_state

        self.screen.fill((0, 0, 0))

        # Board
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                color = (
                    self.COLOR["Dark"]
                    if (row + col) % 2 == 1
                    else self.COLOR["Light"]
                )
                pygame.draw.rect(
                    self.screen, color,
                    (col * self.TILE_SIZE, row * self.TILE_SIZE,
                     self.TILE_SIZE, self.TILE_SIZE)
                )

        # Highlighted actions
        dest_set: set[tuple[int, int]] = set()
        for action in self.highlighted_actions:
            dest_set.add(action[-1])

        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                pawn = state[row][col]
                cx = col * self.TILE_SIZE + self.TILE_SIZE // 2
                cy = row * self.TILE_SIZE + self.TILE_SIZE // 2
                r  = self.TILE_SIZE // 3

                if (row, col) == self.selected_pawn:
                    pygame.draw.rect(
                        self.screen, self.COLOR["Selected"],
                        (col * self.TILE_SIZE, row * self.TILE_SIZE,
                         self.TILE_SIZE, self.TILE_SIZE), 4
                    )

                if (row, col) in dest_set:
                    pygame.draw.rect(
                        self.screen, self.COLOR["Green"],
                        (col * self.TILE_SIZE, row * self.TILE_SIZE,
                         self.TILE_SIZE, self.TILE_SIZE), 3
                    )
                # Pawn
                if pawn != self.EMPTY_TILE:
                    color = (
                        self.COLOR["White"]
                        if pawn in (self.WHITE_PAWN, self.WHITE_KING)
                        else self.COLOR["Black"]
                    )
                    pygame.draw.circle(self.screen, color, (cx, cy), r)

                    if pawn == self.WHITE_KING:
                        pygame.draw.circle(
                            self.screen, self.COLOR["Flare"],
                            (cx, cy), self.TILE_SIZE // 4, 3
                        )
                    elif pawn == self.BLACK_KING:
                        pygame.draw.circle(
                            self.screen, self.COLOR["Bolt"],
                            (cx, cy), self.TILE_SIZE // 4, 3
                        )

        pygame.display.flip()


    def human_input(self) -> tuple[list, float, bool, int]:
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

                legal = self.available_moves(self.current_state, self.human_player)

                if self.selected_pawn is not None:
                    matching = [
                        a for a in legal
                        if a[0] == self.selected_pawn and a[-1] == (row, col)
                    ]
                    if matching:
                        action = matching[0]
                        state, reward, done, player = self.step(action)
                        self.selected_pawn    = None
                        self.highlighted_actions = []
                        return state, reward, done, player
                    else:
                        self.selected_pawn    = None
                        self.highlighted_actions = []
                        
                if self.current_state[row][col] != self.EMPTY_TILE and self._is_own(
                    self.current_state[row][col], self.human_player
                ):
                    origins = [a for a in legal if a[0] == (row, col)]
                    if origins:
                        self.selected_pawn       = (row, col)
                        self.highlighted_actions = origins

        return None, None, self.done, None