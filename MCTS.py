"""
    @file        MCTS.py
    @author      Mowibox (Ousmane THIONGANE)
    @brief       Monte-Carlo Tree Search algorithm
    @version     1.0
    @date        2025-02-02
    
"""
# Imports
import math
import random 
from copy import deepcopy


class MCTSNode:
    """
    Monte-Carlo Tree Search Node
    """
    def __init__(self, state, player, env, c, parent=None, action=None):
        self.state = state
        self.player = player
        self.env = env
        self.parent = parent
        self.action = action
        self.children = []
        self.untried_actions = env.available_moves(self.state, self.player)
        self.n = 0
        self.w = 0
        self.is_terminal = env.check_termination(self.state, simulation=True)[0]
        self.c = c


    @property
    def fully_expanded(self):
        """
        Checks if all possible actions have been tried
        """
        return len(self.untried_actions) == 0
    

    def expand(self):
        """
        Picks an untried action, evaluates it, generates the node for 
        the resulting state (also add it to the children) and returns it.
        """
        action = self.untried_actions.pop()
        next_state, next_player = self.env.transition_function(deepcopy(self.state), action, self.player, simulation=True)

        child_node = MCTSNode(next_state, next_player, self.env, self.c, parent=self, action=action)
        self.children.append(child_node)

        return child_node
    

    def rollout(self):
        """
        Performs a random simulation 
        """
        env = self.env
        state = deepcopy(self.state)
        player = self.player
        done, result = self.env.check_termination(state, simulation=True)

        while not done:
            possible_actions = env.available_moves(state, player)
            if not possible_actions:
                break
            action = random.choice(possible_actions)
            state, player = env.transition_function(deepcopy(state), action, player, simulation=True)
            done, result = env.check_termination(state, simulation=True)

        return result
    

    def backpropagate(self, result):
        """
        Backpropagates the simulation result up to the root
        @param result: The simulation result
        """
        self.n += 1
        if self.parent:
            if result == self.parent.player:
                self.w += 1
            elif result is not None:
                self.w -= 1
            self.parent.backpropagate(result)


    def traverse(self):
        """
        Explore the tree until non-fully expanded node is reached
        """
        node = self

        while node.fully_expanded and not node.is_terminal:
            node = node.best_uct_child()

        if node.is_terminal:
            return node
        
        return node.expand()
    

    def winrate(self):
        """
        Returns the winrate the provided node
        """
        return self.w/self.n if self.n > 0 else 0


    def uct(self):
        """
        Computes the UCT value of the provided node
        """
        return self.winrate() + self.c*math.sqrt(math.log(self.parent.n)/self.n)
    

    def best_child(self):
        """
        Returns the best child based on winrate
        """
        return max(self.children, key=lambda child: child.winrate())
    

    def best_uct_child(self):
        """
        Returns the best child based on UCT value
        """
        return max(self.children, key=lambda child: child.uct())
    

def mcts(state, player, env, c=math.sqrt(2), iters=5000):
    """
    Runs the MCTS algorithm

    @param state: The current state 
    @param player: The current player 
    @param env: The provided environment 
    @param c: The exploration parameter
    @param iters: The number of simulation iterations
    """
    root = MCTSNode(deepcopy(state), player, env, c)
    for _ in range(iters):
        leaf = root.traverse()
        simulation_result = leaf.rollout()
        leaf.backpropagate(simulation_result)
    
    return root.best_child().action, root

        