# Checkers-RL
A reinforcement learning agent capable of solving checkers.

## Overview 

This repository proposes a reinforcement learning-based approach to train an agent capable of playing checkers. The goal is to develop a model that can adapt to the game complexity by using some advanced reinforcement learning algorithms. The project provides tools to train, evaluate and visualize the agent performance, as well as to allow human to play against the trained agent. The environment details are specified in the [documentation](https://github.com/Mowibox/Checkers-RL/wiki/Documentation). 

## Code usage

Download the necessary packages:
```
pip install -r requirements.txt
```

Download the repository:

```bash
git clone https://github.com/Mowibox/CheckersRL.git
```

Run inside the repository:

    python3 main.py [options]

        usage: main.py [-h] [--render] [-t TRAIN] [-e] [--human [HUMAN]]

        options:
            -h, --help            show this help message and exit
            --render              Enable rendering
            -t TRAIN, --train TRAIN
                                    Train the RL model
            -e EVALUATE, --evaluate EVALUATE
                                    Evaluate the provided RL model (Use 'random'/'mcts'/'model
                                    filepath')
            --human [HUMAN]       Allows human to play against computer [w, b] (default: w)

### Commands examples:
Train a TD(λ) LVFA model named `model.pkl`:

    python main.py --train model.pkl

Evaluate a random agent: 

    python main.py --evaluate random

Evaluate a TD(λ) LVFA model: 

    python main.py --evaluate model.pkl

Evaluate a MCTS agent: 

    python main.py --evaluate model.pkl

See the evaluation episode:

    python main.py --evaluate model.pkl --render

Play against the agent (white pawns by default):

    python main.py --evaluate model.pkl --human


## Author 
[Ousmane THIONGANE](https://github.com/Mowibox)

## References 

* [1] Neto, H.C., Julia, R.M.S., Caexeta, G.S. et al. LS-VisionDraughts: improving the performance of an agent for checkers by integrating computational intelligence, reinforcement learning and a powerful search method. Appl Intell 41, 525–550 (2014). https://doi.org/10.1007/s10489-014-0536-y
