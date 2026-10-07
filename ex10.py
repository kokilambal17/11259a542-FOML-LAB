import numpy as np                                       # fast maths
import gymnasium as gym                                  # the game world

game = gym.make("FrozenLake-v1", is_slippery=False)      # the 4x4 ice grid
Q = np.zeros((16, 4))                                    # score-sheet: 16 squares x 4 moves, all 0

for round in range(2000):                                # play 2000 practice games
    state, _ = game.reset()                              # start at the beginning
    done = False                                         # not finished yet
    while not done:                                      # keep moving until game ends
        if np.random.rand() < 0.2:                       # 20% of the time: try a random move (explore)
            move = game.action_space.sample()            # pick a random move
        else:                                            # otherwise: use the best known move
            move = np.argmax(Q[state])                   # best move for this square
        new_state, reward, done, _, _ = game.step(move)  # make the move, see result
        Q[state, move] += 0.8 * (reward + 0.95 * np.max(Q[new_state]) - Q[state, move])  # learn
        state = new_state                                # move to the new square

print("Training done. The agent has learned the path.")
arrows = ["left", "down", "right", "up"]                 # names for the 4 moves
for square in [0, 6, 10, 14]:                            # check a few squares
    best = np.argmax(Q[square])                          # best move for that square
    print("From square", square, "-> best move:", arrows[best])