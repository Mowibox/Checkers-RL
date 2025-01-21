"""
    @file        main.py
    @author      Mowibox (Ousmane THIONGANE)
    @brief       Main file
    @version     1.0
    @date        2025-01-21
    
"""
# Imports 
import pygame
import random
import argparse
from CheckersRL import CheckersRL


def evaluate(env=None, n_episodes=1, render=False, human_play=None):
    """
    Evaluation function

    @param env: The provided CheckersRL environment
    @param n_episodes: The number of episodes
    @param render: Enables rendering
    @param human_play: Enables human player 
    """
    if human_play is not None:
        env = CheckersRL(human_play=human_play)
    else:
        env = CheckersRL()

    rewards = []
    for _ in range(n_episodes):
        total_reward = 0
        done = False
        state, _ = env.reset()
        while not done:
            if human_play is not None or render:
                env.render()
                pygame.time.delay(200)
            if human_play is not None and env.current_player == human_play:
                _, reward, done, _ = env.human_input()
            else:
                available_moves = env.available_moves()
                action = random.choice(available_moves)
                state, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    print(f"Mean reward: {sum(rewards)/len(rewards)}")


def train():
    """
    Training function
    """
    ...


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('--human', nargs='?', const='w')
    args = parser.parse_args()

    if args.train:
        train()
    
    if args.evaluate:
        human_play = None
        if args.human:
            human_play = CheckersRL.WHITE_PAWN if args.human == 'w' else CheckersRL.BLACK_PAWN
        evaluate(render=args.render, human_play=human_play)


if __name__ == "__main__":
    main()
