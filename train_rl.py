import os
import argparse
import pygame
import torch
from torch.utils.tensorboard import SummaryWriter
from board import BOARD_WIDTH, BOARD_HEIGHT, CELL_SIZE, SIDE_PANEL_WIDTH
from rl_env import TetrisEnv
from rl_agent import Agent


def draw_envs(screen, envs, cols=4):
    """Render multiple environments in a grid layout."""
    screen.fill((30, 30, 30))
    rows = (len(envs) + cols - 1) // cols
    for idx, env in enumerate(envs):
        col = idx % cols
        row = idx // cols
        offset_x = col * env_width
        offset_y = row * env_height
        env.render(screen, offset_x=offset_x, offset_y=offset_y)
        # separator
        pygame.draw.rect(screen, (80, 80, 80),
                         pygame.Rect(offset_x, offset_y, env_width, env_height),
                         1)
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--num_envs', type=int, default=4, choices=range(1,9))
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    envs = [TetrisEnv() for _ in range(args.num_envs)]
    state_dim = len(envs[0]._get_obs())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = Agent(state_dim, device=device)
    if args.resume:
        agent.load(args.resume)

    pygame.init()
    global env_width, env_height
    env_width = BOARD_WIDTH * CELL_SIZE + SIDE_PANEL_WIDTH
    env_height = BOARD_HEIGHT * CELL_SIZE
    cols = 4
    rows = (args.num_envs + cols - 1) // cols
    screen = pygame.display.set_mode((env_width * cols, env_height * rows))
    clock = pygame.time.Clock()

    writer = SummaryWriter()

    episode_rewards = [0 for _ in envs]
    for episode in range(args.episodes):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
        for idx, env in enumerate(envs):
            state = env._get_obs()
            action = agent.act(state, epsilon=max(0.05, 1 - episode/500))
            next_state, reward, done, _ = env.step(action)
            agent.push(state, action, reward, next_state, float(done))
            loss = agent.update()
            episode_rewards[idx] += reward
            if done:
                writer.add_scalar(f'env_{idx}/reward', episode_rewards[idx], episode)
                episode_rewards[idx] = 0
                env.reset()
            if loss is not None:
                writer.add_scalar('loss', loss, agent.steps_done)
        draw_envs(screen, envs, cols=cols)
        if episode % 100 == 0 and episode > 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f'ckpt_{episode}.pt')
            agent.save(ckpt_path)
            # keep only the 5 most recent checkpoints
            ckpts = sorted([
                f for f in os.listdir(args.checkpoint_dir)
                if f.startswith('ckpt_') and f.endswith('.pt')
            ])
            while len(ckpts) > 5:
                oldest = ckpts.pop(0)
                os.remove(os.path.join(args.checkpoint_dir, oldest))
        clock.tick(60)

    agent.save(os.path.join(args.checkpoint_dir, 'final.pt'))
    pygame.quit()

if __name__ == '__main__':
    main()

