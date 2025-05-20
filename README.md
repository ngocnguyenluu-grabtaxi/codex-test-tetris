# Self-Playing Tetris

This repository contains a simple console-based Tetris implementation in Python.
The game does not require any external dependencies beyond the Python standard library.

Run the game with:

```
python3 self_playing_tetris.py
```

The game will automatically choose moves based on a heuristic algorithm and display the board in the terminal until it reaches a game-over state.

## Pygame Version

If you have `pygame` installed, you can also watch the autoplayer run in a window
with coloured blocks. Run it with:

```
python3 self_playing_tetris_pygame.py
```

This version uses the same heuristic logic but renders the board using `pygame`
and assigns a different colour to each tetromino.
