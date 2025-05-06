from enum import Enum
import math
from typing import Any, Dict, List, Optional


class NatureAction(Enum):
    SMALL_BLIND = "small_blind"
    BIG_BLIND = "big_blind"
    DEAL_HOLE_CARDS = "deal_hole_cards"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class PlayerAction(Enum):
    CALL = "call"
    RAISE = "raise"
    FOLD = "fold"


class MCTSNodeState(object):
    def __init__(self, hole_cards, community_cards, player_turn):
        self.hole_cards = hole_cards
        self.community_cards = community_cards
        self.player_turn = player_turn
    
    def legal_actions(self):
        return [pa for pa in PlayerAction]



class MCTSNode(object):
    def __init__(self, state: MCTSNodeState, parent: Optional["MCTSNode"] = None,
                 parent_action: Optional[Any] = None):
        self.state = state
        self.parent = parent
        self.action = parent_action 
        self.children = {}
        self.visits = 0
        self.value = 0
        self.is_end_game = False
        self.q_values = {}
        self.visit_action = {}

        for a in state.legal_actions():
            self.q_values[a]      = 0.0
            self.visit_action[a] = 0
    
    def is_fully_expanded(self) -> bool:
        return set(self.children.keys()) == set(self.q_values.keys())

    def is_terminal(self) -> bool:
        return self.state.is_terminal()
    
    def expand(self) -> "MCTSNode":
        untried = [a for a in self.q_values if a not in self.children]
        action = untried.pop()
        next_s = self.state.next_state(action)
        child  = MCTSNode(next_s, parent=self, parent_action=action)
        self.children[action] = child
        return child

    def best_child(self, c_param: float = 1.4) -> "MCTSNode":
        bestscore, best = -1e9, None
        for a, child in self.children.items():
            Q = self.q_values[a]
            Nsa = self.visit_action[a]
            U = c_param * math.sqrt(math.log(self.visits + 1) / (1 + Nsa))
            score = Q + U
            if score > bestscore:
                bestscore, best = score, child
        return best

    def backpropagate(self, reward: float, alpha: float = 0.1, gamma: float = 0.99):
        self.visits += 1
        if self.parent is not None:
            a = self.action
            if not self.is_terminal():
                max_q_next = max(self.q_values.values())
            else:
                max_q_next = 0.0
            old_q = self.parent.q_values[a]
            self.parent.q_values[a] += alpha * (
                reward + gamma * max_q_next - old_q
            )
            self.parent.visit_action[a] += 1
            self.parent.backpropagate(reward, alpha, gamma)

    def __repr__(self):
        return (f"<Node act={self.action} visits={self.visits} "
                f"Q={self.q_values}>")



class MCTSTree(object):
    def __init__(self):
        self.create_tree()

    def create_tree(self):
        root_state = MCTSNodeState(hole_cards=None, community_cards=None, player_turn=0)
        self.root = MCTSNode(state=root_state)
        self.root.children = {
            "small_blind": MCTSNode(state=MCTSNodeState(hole_cards=None, community_cards=None, player_turn=1), parent=self.root),
            "big_blind": MCTSNode(state=MCTSNodeState(hole_cards=None, community_cards=None, player_turn=1), parent=self.root),
        }

    def select(self):
        node = self.root
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            node = node.best_child()
        return node

    def expand(self, node: MCTSNode):
        return node.expand()

    def simulate(self, node: MCTSNode):
        return node.state.rollout()

    def run_iteration(self):
        leaf   = self.select()
        if not leaf.is_terminal():
            leaf = self.expand(leaf)
        reward = self.simulate(leaf)
        leaf.backpropagate(reward)

    def run(self, n_iters: int):
        for _ in range(n_iters):
            self.run_iteration()

    def best_action(self):
        return max(
            self.root.visit_action.items(),
            key=lambda kv: kv[1]
        )[0]
