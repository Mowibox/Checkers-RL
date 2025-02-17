"""
    @file        main.py
    @author      Mowibox (Ousmane THIONGANE)
    @brief       Main file
    @version     1.0
    @date        2025-01-21
    
"""
# Imports 
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame
import random
import argparse
from MCTS import *
from TDLambda_LVFA import *
from CheckersRL import CheckersRL


def evaluate(filename, env=None, n_episodes=1, render=False, human_play=None):
    """
    Evaluation function

    @param filename: The model filename
    @param env: The provided CheckersRL environment
    @param n_episodes: The number of episodes
    @param render: Enable rendering
    @param human_play: Enables human player 
    """
    if human_play is not None:
        env = CheckersRL(human_play=human_play)
    else:
        env = CheckersRL()

    # agent = TDLambda_LVFA.load(filename)

    rewards = []
    for _ in range(n_episodes):
        total_reward = 0
        done = False
        state, player = env.reset()
        while not done:
            if human_play is not None or render:
                env.render()
                pygame.time.delay(300)
            

            if human_play is not None and env.current_player == human_play:
                state, reward, done, player = env.human_input()
            else:
                available_moves = env.available_moves(state, player)
                # action = random.choice(available_moves)
                # action = agent.policy(state)
                action, root = mcts(deepcopy(state), player, env, iters=1000)
                next_state, reward, done, player = env.step(action)
                state = next_state
            if reward is not None:
                total_reward += reward
        rewards.append(total_reward)
    print(f"Mean reward: {sum(rewards)/len(rewards)}")


def train(filename):
    """
    Training function

    @param filename: The model filename
    """
    env = CheckersRL()
    agent = TDLambda_LVFA(env)
    agent.train()
    agent.save(filename)


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true',
                        help="Enable rendering")
    parser.add_argument('-t', '--train',
                        help="Train the RL model")
    parser.add_argument('-e', '--evaluate',
                        help="Evaluate the provided RL model")
    parser.add_argument('--human', nargs='?', const='w',
                        help="Allows human to play against computer [w, b] (default: w)")
    args = parser.parse_args()

    if args.train is not None:
        train(args.train)
    
    if args.evaluate:
        human_play = None
        if args.human:
            human_play = CheckersRL.WHITE_PAWN if args.human == 'w' else CheckersRL.BLACK_PAWN
        evaluate(filename=args.evaluate, render=args.render, human_play=human_play)


if __name__ == "__main__":
    main()
