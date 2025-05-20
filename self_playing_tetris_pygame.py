import pygame
import random
import sys

BOARD_WIDTH = 10
BOARD_HEIGHT = 20
CELL_SIZE = 30
# extra space on the right for showing next/held pieces
SIDE_PANEL_WIDTH = 6 * CELL_SIZE
# Delay after a piece settles (milliseconds)
MOVE_DELAY = 200
# Delay between each row of the falling piece (milliseconds)
FALL_STEP_DELAY = 50

# Tetromino definitions with their rotation states
TETROMINOES = {
    'I': [
        [(0,0), (1,0), (2,0), (3,0)],
        [(0,0), (0,1), (0,2), (0,3)]
    ],
    'O': [
        [(0,0), (1,0), (0,1), (1,1)]
    ],
    'T': [
        [(1,0), (0,1), (1,1), (2,1)],
        [(1,0), (1,1), (2,1), (1,2)],
        [(0,1), (1,1), (2,1), (1,2)],
        [(1,0), (0,1), (1,1), (1,2)]
    ],
    'S': [
        [(1,0), (2,0), (0,1), (1,1)],
        [(0,0), (0,1), (1,1), (1,2)]
    ],
    'Z': [
        [(0,0), (1,0), (1,1), (2,1)],
        [(1,0), (0,1), (1,1), (0,2)]
    ],
    'J': [
        [(0,0), (0,1), (1,1), (2,1)],
        [(0,0), (1,0), (0,1), (0,2)],
        [(0,0), (1,0), (2,0), (2,1)],
        [(1,0), (1,1), (0,2), (1,2)]
    ],
    'L': [
        [(2,0), (0,1), (1,1), (2,1)],
        [(0,0), (0,1), (0,2), (1,2)],
        [(0,0), (1,0), (2,0), (0,1)],
        [(0,0), (1,0), (1,1), (1,2)]
    ]
}

COLORS = {
    'I': (0, 240, 240),
    'O': (240, 240, 0),
    'T': (160, 0, 240),
    'S': (0, 240, 0),
    'Z': (240, 0, 0),
    'J': (0, 0, 240),
    'L': (240, 160, 0)
}

BACKGROUND_COLOR = (30, 30, 30)
GRID_COLOR = (50, 50, 50)


def empty_board():
    return [[None for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]


def check_collision(board, shape, offset_x, offset_y):
    for x, y in shape:
        px = x + offset_x
        py = y + offset_y
        if px < 0 or px >= BOARD_WIDTH or py >= BOARD_HEIGHT:
            return True
        if py >= 0 and board[py][px] is not None:
            return True
    return False


def drop_y(board, shape, x):
    y = 0
    if check_collision(board, shape, x, y):
        return -1
    while not check_collision(board, shape, x, y + 1):
        y += 1
    return y


def place_piece(board, shape, offset_x, offset_y, piece):
    for x, y in shape:
        board[offset_y + y][offset_x + x] = piece


def clear_lines(board):
    new_board = [row for row in board if any(cell is None for cell in row)]
    lines_cleared = BOARD_HEIGHT - len(new_board)
    while len(new_board) < BOARD_HEIGHT:
        new_board.insert(0, [None] * BOARD_WIDTH)
    for y in range(BOARD_HEIGHT):
        board[y] = new_board[y]
    return lines_cleared


def column_heights(board):
    heights = [0] * BOARD_WIDTH
    for x in range(BOARD_WIDTH):
        for y in range(BOARD_HEIGHT):
            if board[y][x] is not None:
                heights[x] = BOARD_HEIGHT - y
                break
    return heights


def count_holes(board):
    holes = 0
    for x in range(BOARD_WIDTH):
        found_block = False
        for y in range(BOARD_HEIGHT):
            if board[y][x] is not None:
                found_block = True
            elif found_block:
                holes += 1
    return holes


def evaluate(board, lines_cleared):
    heights = column_heights(board)
    aggregate_height = sum(heights)
    bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(BOARD_WIDTH - 1))
    holes = count_holes(board)
    return (lines_cleared * 1.0) - (aggregate_height * 0.1) - (holes * 0.5) - (
        bumpiness * 0.3)


def choose_move(board, piece):
    best_score = float('-inf')
    best_move = (None, None)
    for shape in TETROMINOES[piece]:
        width = max(x for x, _ in shape) + 1
        for x in range(BOARD_WIDTH - width + 1):
            y = drop_y(board, shape, x)
            if y < 0:
                continue
            temp = [row[:] for row in board]
            place_piece(temp, shape, x, y, piece)
            lines = clear_lines(temp)
            score = evaluate(temp, lines)
            if score > best_score:
                best_score = score
                best_move = (shape, x)
    return best_move[0], best_move[1], best_score


def draw_board(screen, board, font, score, next_piece, hold_piece):
    screen.fill(BACKGROUND_COLOR)
    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)
            if cell is not None:
                pygame.draw.rect(screen, COLORS[cell], rect)
    panel_x = BOARD_WIDTH * CELL_SIZE + 10
    score_surf = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_surf, (panel_x, 5))

    next_surf = font.render("Next:", True, (255, 255, 255))
    screen.blit(next_surf, (panel_x, 30))
    if next_piece is not None:
        for x, y in TETROMINOES[next_piece][0]:
            rect = pygame.Rect(panel_x + x * CELL_SIZE,
                               50 + y * CELL_SIZE,
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, COLORS[next_piece], rect)

    hold_surf = font.render("Hold:", True, (255, 255, 255))
    screen.blit(hold_surf, (panel_x, 120))
    if hold_piece is not None:
        for x, y in TETROMINOES[hold_piece][0]:
            rect = pygame.Rect(panel_x + x * CELL_SIZE,
                               140 + y * CELL_SIZE,
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, COLORS[hold_piece], rect)


def main():
    pygame.init()
    width = BOARD_WIDTH * CELL_SIZE + SIDE_PANEL_WIDTH
    height = BOARD_HEIGHT * CELL_SIZE
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Self-Playing Tetris")
    font = pygame.font.SysFont(None, 24)

    board = empty_board()
    score = 0
    current_piece = random.choice(list(TETROMINOES.keys()))
    next_piece = random.choice(list(TETROMINOES.keys()))
    hold_piece = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # determine whether to use hold piece
        if hold_piece is None:
            shape_cur, x_cur, score_cur = choose_move(board, current_piece)
            shape_next, x_next, score_next = choose_move(board, next_piece)
            if shape_cur is None and shape_next is None:
                break
            if score_next > score_cur:
                action = 'hold_current'
                shape, x = shape_next, x_next
                piece = next_piece
            else:
                action = 'use_current'
                shape, x = shape_cur, x_cur
                piece = current_piece
        else:
            shape_cur, x_cur, score_cur = choose_move(board, current_piece)
            shape_hold, x_hold, score_hold = choose_move(board, hold_piece)
            if shape_cur is None and shape_hold is None:
                break
            if score_hold > score_cur:
                action = 'use_hold'
                shape, x = shape_hold, x_hold
                piece = hold_piece
            else:
                action = 'use_current'
                shape, x = shape_cur, x_cur
                piece = current_piece

        y = drop_y(board, shape, x)
        if y < 0:
            break

        for step in range(y + 1):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            if not running:
                break
            draw_board(screen, board, font, score, next_piece, hold_piece)
            for px, py in shape:
                rect = pygame.Rect((x + px) * CELL_SIZE,
                                   (step + py) * CELL_SIZE,
                                   CELL_SIZE, CELL_SIZE)
                if step + py >= 0:
                    pygame.draw.rect(screen, COLORS[piece], rect)
            pygame.display.flip()
            pygame.time.wait(FALL_STEP_DELAY)

        if not running:
            break

        place_piece(board, shape, x, y, piece)
        lines = clear_lines(board)
        score += lines

        # update piece order based on the chosen action
        if action == 'hold_current':
            hold_piece = current_piece
            current_piece = next_piece
            next_piece = random.choice(list(TETROMINOES.keys()))
        elif action == 'use_hold':
            hold_piece = current_piece
            current_piece = next_piece
            next_piece = random.choice(list(TETROMINOES.keys()))
        else:  # use_current
            current_piece = next_piece
            next_piece = random.choice(list(TETROMINOES.keys()))

        draw_board(screen, board, font, score, next_piece, hold_piece)
        pygame.display.flip()
        pygame.time.wait(MOVE_DELAY)

    draw_board(screen, board, font, score, next_piece, hold_piece)
    pygame.display.flip()
    pygame.time.wait(2000)
    pygame.quit()
    print("Game over! Final score:", score)


if __name__ == '__main__':
    main()
