import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

from rl_env import ACTIONS
from board import BOARD_WIDTH, BOARD_HEIGHT

class DQN(nn.Module):
    def __init__(self, input_dim, board_h=20, board_w=10, n_actions=len(ACTIONS)):
        super().__init__()
        board_dim = board_h * board_w
        self.board_h = board_h
        self.board_w = board_w
        self.extras_dim = input_dim - board_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        dummy = torch.zeros(1, 1, board_h, board_w)
        conv_out = self.conv(dummy).view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(conv_out + self.extras_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        board = x[:, : self.board_h * self.board_w]
        board = board.view(-1, 1, self.board_h, self.board_w)
        extras = x[:, self.board_h * self.board_w :]
        feat = self.conv(board).view(board.size(0), -1)
        return self.fc(torch.cat([feat, extras], dim=1))

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.int64),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, state_dim, device='cpu', lr=5e-4, gamma=0.99):
        self.device = device
        self.gamma = gamma
        self.policy_net = DQN(state_dim, board_h=BOARD_HEIGHT, board_w=BOARD_WIDTH).to(device)
        self.target_net = DQN(state_dim, board_h=BOARD_HEIGHT, board_w=BOARD_WIDTH).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.steps_done = 0

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(ACTIONS)
        with torch.no_grad():
            state_v = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.policy_net(state_v)
            return int(torch.argmax(qvals).item())

    def push(self, *transition):
        self.buffer.push(transition)

    def update(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        if self.steps_done % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.steps_done += 1
        return loss.item()

    def save(self, path):
        torch.save({
            'model': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps_done
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(data['model'])
        self.target_net.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.steps_done = data.get('steps', 0)

