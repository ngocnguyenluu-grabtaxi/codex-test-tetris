# Self-Playing Tetris

This repository contains a simple Tetris game implemented with `pygame`.
The game can either be played manually or it can play itself using a
basic AI.

Run the game with:

```
python3 tetris_game.py [--manual]
```

Pass the `--manual` flag if you want to control the pieces yourself. If
the flag is omitted, the AI will play automatically. When playing
manually the side panel will display the keyboard controls.

## Reinforcement Learning Agent

A simple PyTorch based DQN agent is provided for learning to play using the
manual control scheme. Train the agent with:

```
python3 train_rl.py --num_envs 8
```

Checkpoints will be saved periodically in the `checkpoints` directory and
training progress can be monitored with TensorBoard.

Evaluate a saved checkpoint with:

```
python3 eval_rl.py --checkpoint checkpoints/ckpt_100.pt
```

