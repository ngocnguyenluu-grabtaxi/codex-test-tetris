from board import BOARD_WIDTH, BOARD_HEIGHT, TETROMINOES, drop_y, place_piece, clear_lines


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
            place_piece(temp, shape, x, y, piece)
            lines = clear_lines(temp)
            score = evaluate(temp, lines)
            if score > best_score:
                best_score = score
                best_move = (shape, x)
    return best_move[0], best_move[1], best_score
