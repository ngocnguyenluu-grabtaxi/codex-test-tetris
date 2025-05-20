BOARD_WIDTH = 10
BOARD_HEIGHT = 20
CELL_SIZE = 30
SIDE_PANEL_WIDTH = 6 * CELL_SIZE

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
