import random
import os
import time

BOARD_WIDTH = 10
BOARD_HEIGHT = 20

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

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def empty_board():
    return [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]

def print_board(board, score):
    clear_screen()
    for row in board:
        print('|' + ''.join('#' if cell else ' ' for cell in row) + '|')
    print('+' + '-' * BOARD_WIDTH + '+')
    print(f'Score: {score}')

def check_collision(board, shape, offset_x, offset_y):
    for x, y in shape:
        px = x + offset_x
        py = y + offset_y
        if px < 0 or px >= BOARD_WIDTH or py >= BOARD_HEIGHT:
            return True
        if py >= 0 and board[py][px]:
            return True
    return False

def drop_y(board, shape, x):
    y = 0
    if check_collision(board, shape, x, y):
        return -1
    while not check_collision(board, shape, x, y+1):
        y += 1
    return y

def place_piece(board, shape, offset_x, offset_y):
    for x, y in shape:
        board[offset_y + y][offset_x + x] = 1

def clear_lines(board):
    new_board = [row for row in board if not all(row)]
    lines_cleared = BOARD_HEIGHT - len(new_board)
    while len(new_board) < BOARD_HEIGHT:
        new_board.insert(0, [0] * BOARD_WIDTH)
    for y in range(BOARD_HEIGHT):
        board[y] = new_board[y]
    return lines_cleared

def column_heights(board):
    heights = [0] * BOARD_WIDTH
    for x in range(BOARD_WIDTH):
        for y in range(BOARD_HEIGHT):
            if board[y][x]:
                heights[x] = BOARD_HEIGHT - y
                break
    return heights

def count_holes(board):
    holes = 0
    for x in range(BOARD_WIDTH):
        found_block = False
        for y in range(BOARD_HEIGHT):
            if board[y][x]:
                found_block = True
            elif found_block:
                holes += 1
    return holes

def evaluate(board, lines_cleared):
    heights = column_heights(board)
    aggregate_height = sum(heights)
    bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(BOARD_WIDTH-1))
    holes = count_holes(board)
    return (lines_cleared * 1.0) - (aggregate_height * 0.1) - (holes * 0.5) - (bumpiness * 0.3)

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
            place_piece(temp, shape, x, y)
            lines = clear_lines(temp)
            score = evaluate(temp, lines)
            if score > best_score:
                best_score = score
                best_move = (shape, x)
    return best_move

def main():
    board = empty_board()
    score = 0
    while True:
        piece = random.choice(list(TETROMINOES.keys()))
        shape, x = choose_move(board, piece)
        if shape is None:
            break
        y = drop_y(board, shape, x)
        if y < 0:
            break
        place_piece(board, shape, x, y)
        lines = clear_lines(board)
        score += lines
        print_board(board, score)
        time.sleep(0.1)
    print('Game over! Final score:', score)

if __name__ == '__main__':
    main()
