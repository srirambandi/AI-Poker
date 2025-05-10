from state_abstraction import StateAbstraction
import random as rand
import math
import json
import os

class MCTSNode:
	def __init__(self, state, parent=None, action=None, isNature=False, isOpponent=False, street="preflop"):
		
		self.state = state      # abstract state
		self.parent = parent    # reference to parent node
		self.action = action    # action that led to this state
		self.children = {}      # dictionary {action: child_node}
		self.visits = 0
		self.value = 0
		
		# booleans indicate whose node it is
		self.isNature = isNature
		self.isOpponent = isOpponent
		
		self.street = street
		
		if parent is None:
			self.raiseCount = 0
			self.streetRaiseCount = 0
		else:
			self.raiseCount = parent.raiseCount
			self.streetRaiseCount = parent.streetRaiseCount

		# increment raise counts we just raised
		if action == "raise":
			self.raiseCount += 1
			self.streetRaiseCount += 1
		
		# set streetRaiseCount to 0 if started a new street
		if parent and parent.street != street:
			self.streetRaiseCount = 0

    # check if we have tried all actions from this state
	def isFullyExpanded(self):
		return len(self.children) == len(self.getValidActions())
	
    # return child w highest UCB score
	def getBestChild(self, explorationWeight=160.0):

		# UCB formula: (value / visits) + explorationWeight * sqrt(2 * ln(parent visits) / child visits)

		if not self.children:
			return None
		
		ucbScores = {}
		for action, child in self.children.items():
			
			exploitation = child.value / child.visits if child.visits > 0 else 0
			exploration = explorationWeight * math.sqrt(2 * math.log(self.visits) / child.visits) if child.visits > 0 else float('inf')
			ucbScores[action] = exploitation + exploration
		
		# return the child with the highest UCB score
		bestAction = max(ucbScores, key=ucbScores.get)
		return self.children[bestAction]



	def getValidActions(self):
		
		# if terminal no valid actions
		if self.isTerminal():
			return []
		
		if self.isNature:
			return ["deal"]
		
		actions = []
		
		# calling & folding always valid
		actions.append("call")
		actions.append("fold")
		
		# check if we are at max raise count
		# NOTE there will be times where we cant raise do to our stack size. Were not checking for that. We might try to raise when we cant.
		if self.raiseCount < 4 and self.streetRaiseCount < 2:
			actions.append("raise")
		
		return actions
	
	def isTerminal(self):
		return self.action == "fold" or self.street == "showdown"
	

	def getReward(self):
		# for terminal nodes
		if self.action == "fold":
			return -1
		# for showndown approximate winning probability based abstract states
		elif self.street == "showdown":
			handStrength = self.getHandStrength()
			return handStrength * 2 - 1
	
    # approximate hand strength based on abstract state
	def getHandStrength(self):
		# higher states have better hand strength
		handRanking = {
			"highCardLow": 0.1, "highCardHigh": 0.15,
			"pairLow": 0.2, "pairHigh": 0.25,
			"twoPairLowLow": 0.3, "twoPairHighLow": 0.35, "twoPairHighHigh": 0.4,
			"threeLow": 0.45, "threeHigh": 0.5,
			"straightLow": 0.55, "straightHigh": 0.6,
			"flush": 0.65,
			"fullHouseLowLow": 0.7, "fullHouseHighLow": 0.75, "fullHouseHighHigh": 0.8,
			"fourLow": 0.85, "fourHigh": 0.9,
			"straightFlush": 0.95,
			"royalFlush": 1.0,
			# it better to have hidden cards if you have shit (ie. CB)
			"communityBestF": 0.05, "communityBestT": 0.04, "communityBestR": 0.03
		}
		
		# for preflop get hand strength based on abstract state (1-8)
		if self.street == "preflop":
			return float(self.state) / 8.0
		
		return handRanking.get(self.state, 0.03)  # default to shit if no state

    # log node in JSON file format
	def toJSON(self, nodeId):
		jsonDict = {
			"id": nodeId,
			"state": self.state,
			"action": self.action,
			"visits": self.visits,
			"value": self.value,
			"isNature": self.isNature,
			"isOpponent": self.isOpponent,
			"street": self.street,
			"raiseCount": self.raiseCount,
			"streetRaiseCount": self.streetRaiseCount,
			"children": {}  # hold child IDs
		}
		return jsonDict

class MCTSTree:
	def __init__(self):
		self.stateAbstractor = StateAbstraction()
		self.root = MCTSNode(state="root", isNature=True)
		self.maxSimulationDepth = 20
	
    # do MCTS
	def search(self, iterations=1000):
		
		for i in range(0, iterations):
			# select a node to expand
			node = self.select(self.root)
			
			# expand the node
			if not node.isTerminal() and node.visits > 0:
				node = self.expand(node)
			
			# simulate from expanded node
			reward = self.simulate(node)
			
			# backprop time
			self.backpropagate(node, reward)
		
		# return best action from root
		if self.root.children:
			bestChild = None
			bestVal = float('-inf')
			
			# get child w highest ratio of value/visits
			for action, child in self.root.children.items():
				if child.visits > 0:
					childValue = child.value / child.visits
					if childValue > bestVal:
						bestVal = childValue
						bestChild = child
			
			return bestChild.action
		
		return "call"  # call if no children
	
    # select node to expand using UCB
	def select(self, node):
		while not node.isTerminal():
			if not node.isFullyExpanded():
				return node
			
			node = node.getBestChild()
			if node is None:
				break
		
		return node
	
	def expand(self, node):
		# expand node, adding child
		possibleActions = node.getValidActions()
		
		# remove actions that already have children
		for action in list(node.children.keys()):
			if action in possibleActions:
				possibleActions.remove(action)
		
		if not possibleActions:
			return node
		
		# choose a random action that hasnt been tried
		action = rand.choice(possibleActions)
		
		# determine the next state and street
		nextState, nextStreet, isNatureNext, isOpponentNext = self.getNextState(node, action)
		
		# create new child node
		childNode = MCTSNode(
			state = nextState, 
			parent = node, 
			action = action, 
			isNature = isNatureNext,
			isOpponent = isOpponentNext,
			street = nextStreet
		)
		
		# add child to parent's children dictionary
		node.children[action] = childNode
		
		return childNode
	
    # get next state & street given current node & action
	def getNextState(self, node, action):
		nextStreet = node.street
		isNatureNext = False
		isOpponentNext = False
		
		# if fold next state is terminal
		if action == "fold":
			return "terminal", "showdown", False, False
		
        # after nature node it is own turn or opp
		if node.isNature:
			# transition to a preflop state
			if node.street == "preflop":
				# after preflop wqe are in numbered abstract state (1-8)
				nextState = str(rand.randint(1, 8))
				isOpponentNext = True

            # transition to a flop state
			elif node.street == "flop":
				possibleFlopStates = [
					"highCardLow", "highCardHigh", "pairLow", "pairHigh",
					"twoPairLowLow", "twoPairHighLow", "twoPairHighHigh",
					"threeLow", "threeHigh", "straightLow", "straightHigh",
					"straightLow-1F", "straightHigh-1F", "flush", "flush-1F", "flush-2",
					"fullHouseLowLow", "fullHouseHighLow", "fullHouseHighHigh",
					"fourLow", "fourHigh", "straightFlush", "straightFlush-1F",
					"royalFlush", "communityBestF"
				]
				nextState = rand.choice(possibleFlopStates)
				isOpponentNext = True

			# transition to a flop state
			elif node.street == "turn":
				possibleTurnStates = [
					"highCardLow", "highCardHigh", "pairLow", "pairHigh",
					"twoPairLowLow", "twoPairHighLow", "twoPairHighHigh",
					"threeLow", "threeHigh", "straightLow", "straightHigh",
					"straightLow-1T", "straightHigh-1T", "flush", "flush-1T",
					"fullHouseLowLow", "fullHouseHighLow", "fullHouseHighHigh",
					"fourLow", "fourHigh", "straightFlush", "straightFlush-1T",
					"royalFlush", "communityBestT"
				]
				nextState = rand.choice(possibleTurnStates)
				isOpponentNext = True

			# transition to a river state
			elif node.street == "river":
				possibleRiverStates = [
					"highCardLow", "highCardHigh", "pairLow", "pairHigh",
					"twoPairLowLow", "twoPairHighLow", "twoPairHighHigh",
					"threeLow", "threeHigh", "straightLow", "straightHigh",
					"flush", "fullHouseLowLow", "fullHouseHighLow", "fullHouseHighHigh",
					"fourLow", "fourHigh", "straightFlush", "royalFlush", 
					"communityBestR"
				]
				nextState = rand.choice(possibleRiverStates)
				isOpponentNext = True

        # our turn after opp
		elif node.isOpponent:
			nextState = node.state  # state doesn't change after opponent move
			isOpponentNext = False
			
        # after we act
		else:
			if action == "call":
				# check to go to next street
				if node.parent and node.parent.isOpponent:
					# if both called go to next street
					if node.street == "preflop":
						nextStreet = "flop"
						isNatureNext = True
					elif node.street == "flop":
						nextStreet = "turn"
						isNatureNext = True
					elif node.street == "turn":
						nextStreet = "river"
						isNatureNext = True
					elif node.street == "river":
						nextStreet = "showdown"
				nextState = node.state
				isOpponentNext = not isNatureNext
				
			elif action == "raise":
				nextState = node.state
				isOpponentNext = True
		
		return nextState, nextStreet, isNatureNext, isOpponentNext
	
    # sim rand movwes from node
	def simulate(self, node):
		currentNode = node
		depth = 0
		
		while currentNode.isTerminal() == False and depth < self.maxSimulationDepth:
			possibleActions = currentNode.getValidActions()
			
			if possibleActions == False:
				break
			
			action = rand.choice(possibleActions)
			nextState, nextStreet, isNatureNext, isOpponentNext = self.getNextState(currentNode, action)
			
			nextNode = MCTSNode(
				state=nextState, 
				parent=currentNode, 
				action=action, 
				isNature=isNatureNext,
				isOpponent=isOpponentNext,
				street=nextStreet
			)
			
			currentNode = nextNode
			depth += 1
		
		# if terminal return reward
		if currentNode.isTerminal():
			return currentNode.getReward()
		
		# else estimate reward
		return currentNode.getHandStrength() * 2 - 1  # scale from [0,1] to [-1,1]
	
    # updates stats for nodes in path
	def backpropagate(self, node, reward):
		while node is not None:
			node.visits += 1
			
			# opp gets negative reward
			if node.isOpponent:
				node.value -= reward
			else:
				node.value += reward
			
			node = node.parent
	
    # get best action from current state
	def getBestAction(self, holeCards, communityCards, street):
		currentState = self.stateAbstractor.get_abstract_state(
			holeCards=holeCards,
			communityCards=communityCards,
			street=street
		)
		
		currentNode = None
		for i, child in self.root.children.items():
			if child.state == currentState and child.street == street:
				currentNode = child
				break
		
		if currentNode is None:
			return "call"
		
		bestAction = None
		bestVal = float('-inf')
		
		for action, child in currentNode.children.items():
			if child.visits > 0:
				childValue = child.value/child.visits
				if childValue > bestVal:
					bestVal = childValue
					bestAction = action
		
		return bestAction if bestAction else "call"

	
    # save tree to JSON
	def savetoJSON(self, filename):
		# create a dictionary to store all nodes
		nodes = {}
		nodeToId = {}
		nextId = 0
		
		# perform a breadth-first traversal to assign IDs and build the JSON structure
		queue = [(self.root, None)]  # (node, parentId)
		while queue:
			node, parentId = queue.pop(0)
			
			# assign an ID to this node
			nodeId = str(nextId)
			nextId += 1
			nodeToId[node] = nodeId
			
			# convert node to JSON structure
			nodes[nodeId] = node.toJSON(nodeId)
			
			# update the parent's children arr
			if parentId is not None:
				parentNode = nodes[parentId]
				parentNode["children"][node.action] = nodeId
			
			# add children to the queue
			for action, childNode in node.children.items():
				queue.append((childNode, nodeId))
		
		# create the final JSON structure
		jsonTree = {
			"root": nodeToId[self.root],
			"simulationDepth": self.maxSimulationDepth,
			"nodes": nodes
		}
		
		# write to JSON file
		with open(filename, 'w') as f:
			json.dump(jsonTree, f, indent=2)
		
		print(f"MCTS tree saved to {filename}")
	
    # load tree from JSON
	def loadFromJson(filename):
		
		# check if file exists
		if not os.path.exists(filename):
			raise FileNotFoundError(f"JSON file {filename} not found")
		
		# load the JSON data
		with open(filename, 'r') as f:
			jsonTree = json.load(f)
		
		# create a new tree
		tree = MCTSTree()
		tree.maxSimulationDepth = jsonTree.get("simulationDepth", 20)
		
		# dictionary to map node IDs to node objects
		nodeDict = {}
		
		# create all nodes without connections
		for nodeId, nodeData in jsonTree["nodes"].items():
			node = MCTSNode(
				state=nodeData["state"],
				action=nodeData["action"],
				isNature=nodeData["isNature"],
				isOpponent=nodeData["isOpponent"],
				street=nodeData["street"]
			)
			node.visits = nodeData["visits"]
			node.value = nodeData["value"]
			node.raiseCount = nodeData["raiseCount"]
			node.streetRaiseCount = nodeData["streetRaiseCount"]
			
			nodeDict[nodeId] = node
		
		# connect nodes
		for nodeId, nodeData in jsonTree["nodes"].items():
			node = nodeDict[nodeId]
			
			# connect children
			for action, childId in nodeData["children"].items():
				childNode = nodeDict[childId]
				childNode.parent = node
				node.children[action] = childNode
		
		# set the root
		tree.root = nodeDict[jsonTree["root"]]
		
		return tree

def trainMCTS(iterations=10000, simulationsPerIteration=100):
	# train an MCTS tree
	tree = MCTSTree()
	
	for i in range(iterations):
		if i % 100 == 0:
			print(f"Training iteration {i}/{iterations}")
		
		# run MCTS search
		tree.search(simulationsPerIteration)
	
	tree.savetoJSON("trained_mcts_tree.json")
	
	return tree

# trainMCTS(iterations=1000, simulationsPerIteration=100)