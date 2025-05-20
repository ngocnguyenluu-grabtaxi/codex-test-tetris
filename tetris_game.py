import pygame
import random
import argparse

from board import (BOARD_WIDTH, BOARD_HEIGHT, CELL_SIZE, SIDE_PANEL_WIDTH,
                   TETROMINOES, COLORS, BACKGROUND_COLOR, GRID_COLOR,
                   empty_board, check_collision, drop_y, place_piece,
                   clear_lines)
import ai

MOVE_DELAY = 200
FALL_STEP_DELAY = 50
ROTATE_STEP_DELAY = 50


def draw_board(screen, board, font, score, next_piece, hold_piece, manual=False, control_font=None):
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

    if manual and control_font:
        controls = [
            "Left/Right: Move",
            "Up: Rotate",
            "Down: Soft drop",
            "Space: Hard drop",
            "C: Hold"
        ]
        for i, text in enumerate(controls):
            surf = control_font.render(text, True, (200, 200, 200))
            screen.blit(surf, (panel_x, 220 + i * 15))


def run_ai(screen, font):
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

        if hold_piece is None:
            shape_cur, x_cur, score_cur = ai.choose_move(board, current_piece)
            shape_next, x_next, score_next = ai.choose_move(board, next_piece)
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
            shape_cur, x_cur, score_cur = ai.choose_move(board, current_piece)
            shape_hold, x_hold, score_hold = ai.choose_move(board, hold_piece)
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

        target_rot = TETROMINOES[piece].index(shape)
        cur_rot = 0
        while running and cur_rot != target_rot:
            cur_rot = (cur_rot + 1) % len(TETROMINOES[piece])
            temp_shape = TETROMINOES[piece][cur_rot]
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            if not running:
                break
            draw_board(screen, board, font, score, next_piece, hold_piece)
            for px, py in temp_shape:
                rect = pygame.Rect((x + px) * CELL_SIZE,
                                   py * CELL_SIZE,
                                   CELL_SIZE, CELL_SIZE)
                if py >= 0 and 0 <= x + px < BOARD_WIDTH:
                    pygame.draw.rect(screen, COLORS[piece], rect)
            pygame.display.flip()
            pygame.time.wait(ROTATE_STEP_DELAY)

        if not running:
            break

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

        if action == 'hold_current':
            hold_piece = current_piece
            current_piece = next_piece
            next_piece = random.choice(list(TETROMINOES.keys()))
        elif action == 'use_hold':
            hold_piece = current_piece
            current_piece = next_piece
            next_piece = random.choice(list(TETROMINOES.keys()))
        else:
            current_piece = next_piece
            next_piece = random.choice(list(TETROMINOES.keys()))

        draw_board(screen, board, font, score, next_piece, hold_piece)
        pygame.display.flip()
        pygame.time.wait(MOVE_DELAY)

    draw_board(screen, board, font, score, next_piece, hold_piece)
    pygame.display.flip()
    pygame.time.wait(2000)
    return score


def run_manual(screen, font, control_font):
    board = empty_board()
    score = 0
    current_piece = random.choice(list(TETROMINOES.keys()))
    next_piece = random.choice(list(TETROMINOES.keys()))
    hold_piece = None
    hold_used = False
    piece_x = BOARD_WIDTH // 2 - 2
    piece_y = 0
    rotation = 0

    fall_timer = pygame.time.get_ticks()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                shape = TETROMINOES[current_piece][rotation]
                if event.key == pygame.K_LEFT:
                    if not check_collision(board, shape, piece_x - 1, piece_y):
                        piece_x -= 1
                elif event.key == pygame.K_RIGHT:
                    if not check_collision(board, shape, piece_x + 1, piece_y):
                        piece_x += 1
                elif event.key == pygame.K_UP:
                    new_rot = (rotation + 1) % len(TETROMINOES[current_piece])
                    new_shape = TETROMINOES[current_piece][new_rot]
                    if not check_collision(board, new_shape, piece_x, piece_y):
                        rotation = new_rot
                elif event.key == pygame.K_SPACE:
                    piece_y = drop_y(board, shape, piece_x)
                elif event.key == pygame.K_c:
                    if not hold_used:
                        if hold_piece is None:
                            hold_piece = current_piece
                            current_piece = next_piece
                            next_piece = random.choice(list(TETROMINOES.keys()))
                        else:
                            hold_piece, current_piece = current_piece, hold_piece
                        piece_x = BOARD_WIDTH // 2 - 2
                        piece_y = 0
                        rotation = 0
                        hold_used = True

        shape = TETROMINOES[current_piece][rotation]
        if pygame.time.get_ticks() - fall_timer > MOVE_DELAY:
            if not check_collision(board, shape, piece_x, piece_y + 1):
                piece_y += 1
            else:
                place_piece(board, shape, piece_x, piece_y, current_piece)
                lines = clear_lines(board)
                score += lines
                current_piece = next_piece
                next_piece = random.choice(list(TETROMINOES.keys()))
                piece_x = BOARD_WIDTH // 2 - 2
                piece_y = 0
                rotation = 0
                hold_used = False
            fall_timer = pygame.time.get_ticks()

        draw_board(screen, board, font, score, next_piece, hold_piece, True, control_font)
        for px, py in shape:
            if piece_y + py >= 0:
                rect = pygame.Rect((piece_x + px) * CELL_SIZE,
                                   (piece_y + py) * CELL_SIZE,
                                   CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, COLORS[current_piece], rect)
        pygame.display.flip()
        pygame.time.wait(30)

    pygame.time.wait(2000)
    return score


def main():
    parser = argparse.ArgumentParser(description="Play Tetris")
    parser.add_argument('--manual', action='store_true', help='Play manually instead of letting the AI play')
    args = parser.parse_args()

    pygame.init()
    width = BOARD_WIDTH * CELL_SIZE + SIDE_PANEL_WIDTH
    height = BOARD_HEIGHT * CELL_SIZE
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Tetris")
    font = pygame.font.SysFont(None, 24)
    small_font = pygame.font.SysFont(None, 18)

    if args.manual:
        score = run_manual(screen, font, small_font)
    else:
        score = run_ai(screen, font)
    pygame.quit()
    print("Game over! Final score:", score)


if __name__ == '__main__':
    main()
