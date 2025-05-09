from pypokerengine.engine.card import Card
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.players import BasePokerPlayer
from MCTSTree import MCTSTree
from state_abstraction import StateAbstraction


class AI13Player(BasePokerPlayer):
	def __init__(self, treeFile="trained_mcts_tree2000.json"):
		# initialize hand counter to track the number of hands played
		self.handCount = 0
		self.stateAbstractor = StateAbstraction()
		self.treeFile = treeFile
		self.tree = self.load_tree()
		self.hand_evaluator = HandEvaluator()

		# track raises per street and total raises for raise limits
		self.raiseCount = 0
		self.streetRaiseCount = 0
		self.lastStreet = None

	def load_tree(self):
		tree = MCTSTree.loadFromJson(self.treeFile)
		return tree

	def declare_action(self, validActions, holeCard, roundState):
		# increment the hand count
		self.handCount += 1

		# check if we moved to a new street and reset street raise count if so
		currentStreet = roundState["street"]
		if self.lastStreet != currentStreet:
			self.streetRaiseCount = 0
			self.lastStreet = currentStreet

		# get abstract state based on current game state
		communityCards = roundState["community_card"]

		# convert validActions to a dictionary for easier access
		validActionDict = {}
		for action in validActions:
			validActionDict[action["action"]] = action
		
		# if using pre-trained tree, get best action from the tree
		bestAction = self.tree.getBestAction(holeCard, communityCards, currentStreet)
		
		# if best action is available, return it
		if bestAction in validActionDict:
			# update raise counts if the action is raise
			if bestAction == "raise":
				if self.raiseCount < 4 and self.streetRaiseCount < 2:
					self.raiseCount += 1
					self.streetRaiseCount += 1
					return bestAction
				else:
					# if we can't raise anymore, call instead
					return "call"
			return bestAction
		else:
			# if no pre-trained tree or best action not available, use a simple policy
			# use built-in hand evaluator to evaluate the hand and make a decision
			communityCards = [Card.from_str(str_card) for str_card in roundState["community_card"]]
			holeCards = [Card.from_str(str_card) for str_card in holeCard]

			reward = self.hand_evaluator.eval_hand(holeCards, communityCards)
            # Raise if above threshold
			if reward > 52000:
				for i in validActions:
					if i["action"] == "raise":
						action = i["action"]
						return action
					action = validActions[1]["action"]
				return action
            # fold
			else:
				for i in validActions:
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
