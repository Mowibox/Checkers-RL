# Checkers-RL

A reinforcement learning agent capable of solving checkers.

![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement%20Learning-✔︎-brightgreen?)
![Issues](https://img.shields.io/github/issues/Mowibox/Checkers-RL)

<p align="center">
  <img src=https://github.com/user-attachments/assets/ad665ed8-6272-40bf-997f-f7facd41e5c7 alt="checkers_env">
  <br>
</p>

## Table of contents

| Section                               | Description                                                      |
| ------------------------------------- | ---------------------------------------------------------------- |
| [Project overview](#project-overview) | General description of the reinforcement learning checkers project |
| [Author](#author)                     | Main contributors information                                     |
| [Documentation](#documentation)       | Links to detailed documentation and presentation materials                |
| [How to use](#how-to-use)             | Instructions for installation and command-line usage             |
| [Contributions](#contributions)       | How to contribute to the repository                              |
| [References](#references)             | Scientific references                          |
| [License](#license)                   | Licensing information                                            |

## Project overview

This repository proposes a reinforcement learning-based approach to train an agent capable of playing checkers. The goal is to develop a model that can adapt to the game complexity by using some advanced reinforcement learning algorithms. The project provides tools to train, evaluate and visualize the agent performance, as well as to allow human to play against the trained agent [[1]](#references).

## Author

| |
| :---: |
| <img src="https://github.com/Mowibox.png" width="100"> |
| [@Mowibox](https://mowibox.github.io)<br>Ousmane THIONGANE |

## Documentation

The environment details are specified in the [documentation wiki](https://github.com/Mowibox/Checkers-RL/wiki/Documentation). A short presentation of the RL approaches used is also available in the ['docs/'](https://github.com/Mowibox/Checkers-RL/tree/main/docs) folder.

## How to use

Download the repository:

```bash
git clone https://github.com/Mowibox/CheckersRL.git
```

Download the necessary packages:

```bash
pip install -r requirements.txt
```

Run inside the repository:

```bash
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
```

### Command-line examples

Train a TD(λ) Linear Value Function Approximation model named `model.pkl`:

```bash
python main.py --train model.pkl
```

Evaluate a random agent:

```bash
python main.py --evaluate random
```

Evaluate a TD(λ) LVFA model:

```bash
python main.py --evaluate model.pkl
```

Evaluate a Monte-Carlo Tree Search (MCTS) agent:

```bash
python main.py --evaluate mcts
```

See the evaluation episode:

```bash
python main.py --evaluate model.pkl --render
```

Play against the agent (white pawns by default):

```bash
python main.py --evaluate model.pkl --human
```

## Contributions

Contributions are always welcome!

* **Report Issues:** Found a bug or have a feature request? Create a new issue [here.](https://github.com/Mowibox/Checkers-RL/issues/new/choose)
* **Fix Bugs & Add Features:** Find out where you can lend a hand by checking out [existing issues.](https://github.com/Mowibox/Checkers-RL/issues)

## References

> * [1] Neto, H.C., Julia, R.M.S., Caexeta, G.S. et al. LS-VisionDraughts: improving the performance of an agent for checkers by integrating computational intelligence, reinforcement learning and a powerful search method. Appl Intell 41, 525–550 (2014). https://doi.org/10.1007/s10489-014-0536-y

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/Mowibox/Checkers-RL/blob/main/LICENSE) file for more details.
