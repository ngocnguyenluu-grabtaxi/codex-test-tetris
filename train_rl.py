import os
import argparse
import pygame
import torch
from torch.utils.tensorboard import SummaryWriter
from board import BOARD_WIDTH, BOARD_HEIGHT, CELL_SIZE, SIDE_PANEL_WIDTH
from rl_env import TetrisEnv
from rl_agent import Agent


def draw_envs(screen, envs, font, *, games_played=0, max_score=0, max_game=0,
              cols=4, scale=0.5):
    """Render multiple environments in a scaled grid layout."""
    screen.fill((30, 30, 30))
    rows = (len(envs) + cols - 1) // cols
    env_w = BOARD_WIDTH * CELL_SIZE + SIDE_PANEL_WIDTH
    env_h = BOARD_HEIGHT * CELL_SIZE
    scaled_w = int(env_w * scale)
    scaled_h = int(env_h * scale)
    for idx, env in enumerate(envs):
        col = idx % cols
        row = idx // cols
        temp = pygame.Surface((env_w, env_h))
        env.render(temp)
        temp = pygame.transform.smoothscale(temp, (scaled_w, scaled_h))
        offset_x = col * scaled_w
        offset_y = row * scaled_h
        screen.blit(temp, (offset_x, offset_y))
        pygame.draw.rect(screen, (80, 80, 80),
                         pygame.Rect(offset_x, offset_y, scaled_w, scaled_h), 1)
    stats = font.render(
        f"Games: {games_played}  Max: {max_score} (Game {max_game})",
        True, (255, 255, 255))
    screen.blit(stats, (5, rows * scaled_h + 5))
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--num_envs', type=int, default=8, choices=range(1,9))
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    envs = [TetrisEnv() for _ in range(args.num_envs)]
    state_dim = len(envs[0]._get_obs())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = Agent(state_dim, device=device)
    if args.resume:
        agent.load(args.resume)

    pygame.init()
    scale = 0.5
    cols = 4
    rows = (args.num_envs + cols - 1) // cols
    env_w = BOARD_WIDTH * CELL_SIZE + SIDE_PANEL_WIDTH
    env_h = BOARD_HEIGHT * CELL_SIZE
    scaled_w = int(env_w * scale)
    scaled_h = int(env_h * scale)
    screen = pygame.display.set_mode((scaled_w * cols, scaled_h * rows + 30))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 18)

    writer = SummaryWriter()

    episode_rewards = [0 for _ in envs]
    games_played = 0
    max_score = 0
    max_game = 0
    running_loss = 0.0
    loss_count = 0
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
                games_played += 1
                if env.score > max_score:
                    max_score = env.score
                    max_game = games_played
                episode_rewards[idx] = 0
                env.reset()
            if loss is not None:
                writer.add_scalar('loss', loss, agent.steps_done)
                running_loss += loss
                loss_count += 1
                if agent.steps_done % 50 == 0:
                    avg = running_loss / loss_count if loss_count else 0
                    print(f"Step {agent.steps_done}: avg loss {avg:.4f}")
                    running_loss = 0.0
                    loss_count = 0
        draw_envs(screen, envs, font, games_played=games_played,
                  max_score=max_score, max_game=max_game, cols=cols, scale=scale)
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

