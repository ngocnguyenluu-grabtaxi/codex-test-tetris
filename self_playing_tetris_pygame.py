import pygame
import random
import sys

BOARD_WIDTH = 10
BOARD_HEIGHT = 20
CELL_SIZE = 30
# Delay between pieces in milliseconds so the AI's moves are visible
MOVE_DELAY = 500

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
    return best_move


def draw_board(screen, board, font, score):
    screen.fill(BACKGROUND_COLOR)
    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)
            if cell is not None:
                pygame.draw.rect(screen, COLORS[cell], rect)
    score_surf = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_surf, (5, BOARD_HEIGHT * CELL_SIZE + 5))


def main():
    pygame.init()
    width = BOARD_WIDTH * CELL_SIZE
    height = BOARD_HEIGHT * CELL_SIZE + 30
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Self-Playing Tetris")
    font = pygame.font.SysFont(None, 24)

    board = empty_board()
    score = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        piece = random.choice(list(TETROMINOES.keys()))
        shape, x = choose_move(board, piece)
        if shape is None:
            break
        y = drop_y(board, shape, x)
        if y < 0:
            break
        place_piece(board, shape, x, y, piece)
        lines = clear_lines(board)
        score += lines
        draw_board(screen, board, font, score)
        pygame.display.flip()
        pygame.time.wait(MOVE_DELAY)

    draw_board(screen, board, font, score)
    pygame.display.flip()
    pygame.time.wait(2000)
    pygame.quit()
    print("Game over! Final score:", score)


if __name__ == '__main__':
    main()
