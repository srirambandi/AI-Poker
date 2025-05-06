import json

from pypokerengine.engine.card import Card
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.players import BasePokerPlayer

from state_abtraction import StateAbtraction

class AI13Player(BasePokerPlayer):

    def __init__(self, policy_file="AI13_Policy.json"):
        super().__init__()
        self.state_abstraction = StateAbtraction()
        self.policy_file = policy_file
        self.policy = self.load_policy()

    def get_state(self, hole_cards, community_cards, street):
        state = self.state_abstraction.getAbstractState(hole_cards, community_cards, street)
        return state
    
    def load_policy(self):
        with open(self.policy_file, 'r') as f:
            state_actions = json.load(f)
        return state_actions

    def declare_action(self, valid_actions, hole_card, round_state):
        # check if we have recorded a valid state for this hand
        state = self.get_state(hole_card, round_state["community_card"], round_state["street"])
        # check if we have a policy for this state and if so, return it
        if state in self.policy.keys():
            self.policy_found += 1
            return self.policy[state]
        # proceed with the hand evaluator
        else:
            self.policy_missed += 1

            community_cards = [Card.from_str(str_card) for str_card in round_state["community_card"]]
            hole_cards = [Card.from_str(str_card) for str_card in hole_card]

            reward = self.hand_evaluator.eval_hand(hole_cards, community_cards)
            # Raise if above threshold
            if reward > 52000:
                for i in valid_actions:
                    if i["action"] == "raise":
                        action = i["action"]
                        return action
                action = valid_actions[1]["action"]
                return action
            # Just fold
            else:
                for i in valid_actions:
                    if i["action"] == "fold":
                        action = i["action"]
                        return action
        
    
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return AI13Player()