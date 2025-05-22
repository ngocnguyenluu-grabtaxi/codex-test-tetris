import random
import pygame
from ai import column_heights, count_holes
from board import (
    BOARD_WIDTH, BOARD_HEIGHT, CELL_SIZE, SIDE_PANEL_WIDTH,
    TETROMINOES, COLORS, BACKGROUND_COLOR, GRID_COLOR,
    empty_board, check_collision, drop_y, place_piece, clear_lines
)

ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_ROTATE = 2
ACTION_SOFT_DROP = 3
ACTION_HARD_DROP = 4
ACTION_HOLD = 5
ACTION_NONE = 6

ACTIONS = [ACTION_LEFT, ACTION_RIGHT, ACTION_ROTATE, ACTION_SOFT_DROP,
           ACTION_HARD_DROP, ACTION_HOLD, ACTION_NONE]

class TetrisEnv:
    """Simple Tetris environment suitable for RL."""
    def __init__(self):
        self.board = None
        self.score = 0
        self.hold_piece = None
        self.hold_used = False
        self.current_piece = None
        self.next_piece = None
        self.piece_x = 0
        self.piece_y = 0
        self.rotation = 0
        self.reset()

    def reset(self):
        self.board = empty_board()
        self.score = 0
        self.hold_piece = None
        self.hold_used = False
        self.current_piece = random.choice(list(TETROMINOES.keys()))
        self.next_piece = random.choice(list(TETROMINOES.keys()))
        self.piece_x = BOARD_WIDTH // 2 - 2
        self.piece_y = 0
        self.rotation = 0
        return self._get_obs()

    def _get_obs(self):
        flat = []
        for row in self.board:
            for cell in row:
                val = 0.0 if cell is None else (list(TETROMINOES.keys()).index(cell) + 1) / 7.0
                flat.append(val)
        # normalize extras
        extras = [list(TETROMINOES.keys()).index(self.current_piece)/7.0,
                  self.rotation/4.0,
                  self.piece_x/BOARD_WIDTH,
                  self.piece_y/BOARD_HEIGHT]
        flat.extend(extras)
        return flat

    def step(self, action):
        reward = 0
        done = False
        shape = TETROMINOES[self.current_piece][self.rotation]

        if action == ACTION_LEFT:
            if not check_collision(self.board, shape, self.piece_x-1, self.piece_y):
                self.piece_x -= 1
        elif action == ACTION_RIGHT:
            if not check_collision(self.board, shape, self.piece_x+1, self.piece_y):
                self.piece_x += 1
        elif action == ACTION_ROTATE:
            new_rot = (self.rotation + 1) % len(TETROMINOES[self.current_piece])
            new_shape = TETROMINOES[self.current_piece][new_rot]
            if not check_collision(self.board, new_shape, self.piece_x, self.piece_y):
                self.rotation = new_rot
                shape = new_shape
        elif action == ACTION_HOLD:
            if not self.hold_used:
                if self.hold_piece is None:
                    self.hold_piece = self.current_piece
                    self.current_piece = self.next_piece
                    self.next_piece = random.choice(list(TETROMINOES.keys()))
                else:
                    self.hold_piece, self.current_piece = self.current_piece, self.hold_piece
                self.piece_x = BOARD_WIDTH // 2 - 2
                self.piece_y = 0
                self.rotation = 0
                self.hold_used = True
                shape = TETROMINOES[self.current_piece][self.rotation]
        if action == ACTION_HARD_DROP:
            self.piece_y = drop_y(self.board, shape, self.piece_x)
        else:
            if action == ACTION_SOFT_DROP:
                if not check_collision(self.board, shape, self.piece_x, self.piece_y+1):
                    self.piece_y += 1
            # gravity
            if not check_collision(self.board, shape, self.piece_x, self.piece_y+1):
                self.piece_y += 1
            else:
                # place piece
                place_piece(self.board, shape, self.piece_x, self.piece_y, self.current_piece)
                lines = clear_lines(self.board)
                self.score += lines
                # reward shaping encourages line clears and penalises
                # messy boards without overwhelming the value targets
                reward += lines * 20
                heights = column_heights(self.board)
                reward -= 0.02 * sum(heights)
                reward -= 0.5 * count_holes(self.board)
                self.current_piece = self.next_piece
                self.next_piece = random.choice(list(TETROMINOES.keys()))
                self.piece_x = BOARD_WIDTH // 2 - 2
                self.piece_y = 0
                self.rotation = 0
                self.hold_used = False
                shape = TETROMINOES[self.current_piece][self.rotation]
                if check_collision(self.board, shape, self.piece_x, self.piece_y):
                    done = True

        obs = self._get_obs()
        if done:
            reward -= 5
        return obs, reward, done, {}

    def render(self, surface, offset_x=0, offset_y=0):
        """Render the environment to the given surface."""
        # draw board background
        board_width_px = BOARD_WIDTH * CELL_SIZE
        board_height_px = BOARD_HEIGHT * CELL_SIZE
        pygame.draw.rect(surface, BACKGROUND_COLOR,
                         pygame.Rect(offset_x, offset_y,
                                    board_width_px + SIDE_PANEL_WIDTH,
                                    board_height_px))

        # draw board cells
        for y, row in enumerate(self.board):
            for x, cell in enumerate(row):
                rect = pygame.Rect(offset_x + x*CELL_SIZE,
                                   offset_y + y*CELL_SIZE,
                                   CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surface, GRID_COLOR, rect, 1)
                if cell is not None:
                    pygame.draw.rect(surface, COLORS[cell], rect)

        # draw current falling piece
        shape = TETROMINOES[self.current_piece][self.rotation]
        for px, py in shape:
            py = self.piece_y + py
            px = self.piece_x + px
            if py >= 0:
                rect = pygame.Rect(offset_x + px*CELL_SIZE,
                                   offset_y + py*CELL_SIZE,
                                   CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surface, COLORS[self.current_piece], rect)

        panel_x = offset_x + board_width_px + 10
        # next piece
        for px, py in TETROMINOES[self.next_piece][0]:
            rect = pygame.Rect(panel_x + px*CELL_SIZE,
                               offset_y + 20 + py*CELL_SIZE,
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, COLORS[self.next_piece], rect)

        # hold piece
        if self.hold_piece is not None:
            for px, py in TETROMINOES[self.hold_piece][0]:
                rect = pygame.Rect(panel_x + px*CELL_SIZE,
                                   offset_y + 90 + py*CELL_SIZE,
                                   CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surface, COLORS[self.hold_piece], rect)


