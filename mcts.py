from enum import Enum
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.hand_evaluator import HandEvaluator
from state_abtraction import StateAbtraction
from pypokerengine.engine.seats import Seats as seats
import math


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

class MCTSNodeState:
    def __init__(self, game_state, player_turn):
        self.game_state = game_state
        self.player_turn = player_turn
        self.hole_cards = self._get_hole_cards()
        self.community_cards = self._get_community_cards()
        self.street = self._get_street()
        self.abstraction = StateAbtraction()
        self.abstract_state = self.abstraction.getAbstractState(
            self.hole_cards, self.community_cards, self.street
        )

    def _get_hole_cards(self):
        player = self.game_state['table'].seats.players[self.player_turn]
        hole_cards = getattr(player, 'holecard', [])
        return [str(card) for card in hole_cards]

    def _get_community_cards(self):
        community_cards = self.game_state.get('community_card', [])
        return [str(card) for card in community_cards]
        
    def _get_street(self):
        mapping = {
            0: NatureAction.DEAL_HOLE_CARDS,
            1: NatureAction.FLOP,
            2: NatureAction.TURN,
            3: NatureAction.RIVER
        }
        return mapping.get(self.game_state['street'], NatureAction.RIVER).value

    def is_terminal(self):
        return  (len(self.game_state['table'].seats.players) <= 1)or (self.game_state['street'] > Const.Street.RIVER)

    def legal_actions(self):
        next = self.game_state['next_player']
        valid = self.game_state['table'].seats.players[next].valid_actions
        return [(v['action'], v.get('amount', 0)) for v in valid]


class MCTSNode(object):
    def __init__(self, state: MCTSNodeState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0
        self.is_end_game = False
    
    def ucb(self, child):
        if child.visit == 0:
            return float('inf')
        else:
            return math.sqrt(math.log(self.visit) / child.visit)


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

class MCTS:
    def __init__(self, player_turn, n_iters=10000):
        self.n_iters = n_iters
        self.player_turn = player_turn
        self.hand_evaluator = HandEvaluator()
        self.emulator = Emulator()
        self.emulator.set_game_rule(
            player_num=2,
            max_round=10,
            small_blind_amount=Const.Action.SMALL_BLIND,
            ante_amount=0
        )

    def search(self, root_game_state):
        root = MCTSNode(MCTSNodeState(root_game_state, self.player_turn))
        for i in range(self.n_iters):
            node = self._select(root)
            reward = self._simulate(node)
            self._backpropagate(node, reward)
        best_child = max(root.children, key=lambda n: n.visits)
        return best_child.action
    
    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _simulate(self, node: MCTSNode):
        pass

    def _select(self, node: MCTSNode):
        pass
    
if __name__ == "__main__":
    emulator = Emulator()
    emulator.set_game_rule(player_num=2, max_round=10, small_blind_amount=10, ante_amount=0)
    players_info = {
        "uuid-1": {"name": "hero",   "stack": 1000}, 
        "uuid-0": {"name": "villain","stack": 1000},  
    }
    init_state = emulator.generate_initial_game_state(players_info)
    game_state, _ = emulator.start_new_round(init_state)
    mcts = MCTS(player_turn=1, n_iters=2000)
    move = mcts.search(game_state)
    print("Best action:", move)