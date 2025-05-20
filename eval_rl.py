import argparse
import pygame
import torch
from board import BOARD_WIDTH, BOARD_HEIGHT, CELL_SIZE
from rl_env import TetrisEnv
from rl_agent import Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()

    env = TetrisEnv()
    state_dim = len(env._get_obs())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = Agent(state_dim, device=device)
    agent.load(args.checkpoint)

    pygame.init()
    screen = pygame.display.set_mode((BOARD_WIDTH*CELL_SIZE, BOARD_HEIGHT*CELL_SIZE))
    clock = pygame.time.Clock()
    state = env.reset()
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        action = agent.act(state, epsilon=0.0)
        state, _, done, _ = env.step(action)
        screen.fill((30,30,30))
        env.render(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == '__main__':
    main()

