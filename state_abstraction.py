from time import sleep
import random as rand

class StateAbstraction(object):

    def __init__(self):
        # sum card values to determine if we have high, mid, or low cards
        self.valueDict = {
            "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
            "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14
        }

    #  take hole cards and return one of 8 abstract state buckets
    def pre_flop_abstraction(self, holeCards):
        PRE_FLOP_CHART = [
            # A   K   Q   J   T   9   8   7   6   5   4  3  2
            [ 8,  8,  8,  8,  8,  7,  6,  5,  4,  4,  4, 4, 4], #A
            [ 8,  8,  8,  8,  8,  5,  4,  3,  3,  3,  2, 2, 2], #K
            [ 8,  6,  8,  8,  8,  5,  3,  2,  2,  2,  2, 2, 2], #Q
            [ 8,  5,  4,  8,  8,  5,  3,  2,  2,  2,  2, 2, 2], #J
            [ 6,  4,  3,  3,  8,  6,  3,  2,  2,  2,  2, 2, 2], #T
            [ 4,  1,  1,  1,  1,  8,  4,  2,  2,  2,  2, 2, 2], #9
            [ 4,  1,  1,  1,  1,  1,  8,  3,  2,  2,  2, 2, 2], #8
            [ 3,  1,  1,  1,  1,  1,  1,  8,  2,  2,  2, 2, 2], #7
            [ 3,  1,  1,  1,  1,  1,  1,  1,  7,  2,  2, 2, 2], #6
            [ 3,  1,  1,  1,  1,  1,  1,  1,  1,  7,  2, 2, 2], #5
            [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  6, 2, 2], #4
            [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 5, 2], #3
            [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1, 4]  #2
        ]

        # assign ranks to cards
        RANK_ORDER = {
            "A": 0, "K": 1, "Q": 2, "J": 3, "T": 4, "9": 5,
            "8": 6, "7": 7, "6": 8, "5": 9, "4": 10, "3": 11, "2": 12
        }

        rank1 = holeCards[0][1]
        rank2 = holeCards[1][1]
        val1 = self.valueDict[rank1]
        val2 = self.valueDict[rank2]

        suited = False
        if holeCards[0][0] == holeCards[1][0]:  # Check if same suit
            suited = True

        if rank1 == rank2:  # Pairs
            row = RANK_ORDER[rank1]
            col = RANK_ORDER[rank2]
            return str(PRE_FLOP_CHART[row][col])
        elif suited:
            if val1 >= val2:
                row = RANK_ORDER[rank1]
                col = RANK_ORDER[rank2]
            else:
                row = RANK_ORDER[rank2]
                col = RANK_ORDER[rank1]
            return str(PRE_FLOP_CHART[row][col])
        else:
            if val1 <= val2:
                row = RANK_ORDER[rank1]
                col = RANK_ORDER[rank2]
            else:
                row = RANK_ORDER[rank2]
                col = RANK_ORDER[rank1]
            return str(PRE_FLOP_CHART[row][col])

    # for pairs, three of a kind & four of a kind
    # 1 if it exists
    # 0 if not
    def have_of_a_kind(self, cards, quantity):
        if len(cards) < quantity:
            return (0, None)
        
        for i in range(len(cards)-(quantity-1)):
            count = 1
            for j in range(i+1, len(cards)):
                if cards[i][1] == cards[j][1]:
                    count += 1
            if count >= quantity:
                if self.valueDict[cards[i][1]] >= 10:
                    return (1, 'high')
                else:
                    return (1, 'low')
        # if we don't have the required quantity of cards, return 0
        return (0, None)

    # 1 if we have two pair
    # -1 if one card away from two pair, -2 if two cards away, -3 if three cards away
    def have_two_pair(self, cards):
        # if we have two cards (preflop) we are always two cards away from two pair
        if len(cards) < 4:
            highCount = 0
            for card in cards:
                if self.valueDict[card[1]] >= 10:
                    highCount += 1

            if highCount > 1:
                return (-2, "highHigh")
            elif highCount == 1:
                return (-2, "highLow")
            else:
                return (-2, "lowLow")
            
        # track rank of each pair
        pairsRank = []

        # count occurrences of each rank
        rankCount = {}
        for card in cards:
            rank = card[1]
            if rank in rankCount:
                rankCount[rank] += 1
                if rankCount[rank] == 2:  # only add when it becomes a pair
                    pairsRank.append(self.valueDict[rank])
            else:
                rankCount[rank] = 1
            
        # count how many pairs we have
        pairCount = sum(1 for count in rankCount.values() if count >= 2)

        if pairCount >= 2:
            highPairCount = 0
            for rank in pairsRank:
                if rank >= 10:
                    highPairCount += 1
            if highPairCount > 1:
                return (1, "highHigh")
            elif highPairCount == 1:
                return (1, "highLow")
            else:
                return (1, "lowLow")
        elif pairCount == 1:
            if pairsRank[0] >= 10:
                return (-1, "highHigh")
            else:
                return (-1, "highLow")
        else:
            return (-2, "lowLow")  # we need two pairs

    # 1 if we have a straight
    # negative number representing how many cards away we are from making a straight
    def have_straight(self, cards):
        if len(cards) < 5:
            return (-5, None)  # we need at least 5 cards to make a straight
        
        # extract unique ranks
        uniqueRanks = set()
        for card in cards:
            uniqueRanks.add(card[1])
        
        # check for Ace low straight (A-2-3-4-5)
        aceLowStraight = {"A", "2", "3", "4", "5"}
        missingForAceLow = aceLowStraight - uniqueRanks
        aceLowDistance = len(missingForAceLow)
        
        # sort cards by value
        uniqueValues = [self.valueDict[rank] for rank in uniqueRanks]
        uniqueValues.sort()
        
        # find longest consecutive sequence
        maxLength = 1
        currentLength = 1
        highCard = 0
        
        # convert to list to handle potential gaps
        values = list(uniqueValues)
        
        for i in range(1, len(values)):
            if values[i] == values[i-1] + 1:
                currentLength += 1
                highCard = max(highCard, values[i])
                maxLength = max(maxLength, currentLength)
            elif values[i] > values[i-1] + 1:
                # Found a gap
                currentLength = 1
                highCard = values[i]
        
        # Special check for Ace-low straight
        if aceLowDistance == 0:
            return (1, "low")  # A-2-3-4-5 is a low straight
        
        if maxLength >= 5:
            # Determine if it's a high straight
            if highCard >= 10:
                return (1, "high")
            else:
                return (1, "low")
        
        # Find distance to straight (cards needed)
        distance = 5 - maxLength
        
        # Return the minimum distance and classify as high or low
        if min(distance, aceLowDistance) == distance:
            if highCard >= 10:
                return (-distance, "high")
            else:
                return (-distance, "low")
        else:
            # Ace-low straight is typically considered low
            return (-aceLowDistance, "low")

    # 1 if we have a flush
    # negative number representing how many cards away we are from making a flush
    def have_flush(self, cards):
        if len(cards) < 5:
            return (-5, None)  # we need at least 5 cards to make a flush
        
        suits = ["H", "D", "C", "S"]
        # count number of cards of each suit
        maxCount = 0
        bestSuit = None
        
        for suit in suits:
            count = 0
            for card in cards:
                if card[0] == suit:
                    count += 1
            if count > maxCount:
                maxCount = count
                bestSuit = suit
        
        if maxCount >= 5:
            # Determine if it's a high flush
            highCardCount = 0
            for card in cards:
                if card[0] == bestSuit and self.valueDict[card[1]] >= 10:
                    highCardCount += 1
            
            if highCardCount > 0:
                return (1, "high")
            else:
                return (1, "low")
        else:
            # Return negative distance
            return (-(5 - maxCount), None)  # flush quality not determined if we don't have it yet

    # 1 if we have a full house
    # negative number representing how many cards away we are from making a full house
    def have_full_house(self, cards):
        if len(cards) < 5:
            return (-5, None)  # we need at least 5 cards to make a full house
        
        # count occurrences of each rank
        rankCount = {}
        for card in cards:
            rank = card[1]
            if rank in rankCount:
                rankCount[rank] += 1
            else:
                rankCount[rank] = 1
        
        # sort ranks by count (descending)
        sortedRanks = sorted(rankCount.items(), key=lambda x: (-x[1], -self.valueDict[x[0]]))
        
        if len(sortedRanks) >= 2 and sortedRanks[0][1] >= 3 and sortedRanks[1][1] >= 2:
            # We have a full house, determine if high or low
            threeRank = sortedRanks[0][0]
            pairRank = sortedRanks[1][0]
            
            threeHigh = self.valueDict[threeRank] >= 10
            pairHigh = self.valueDict[pairRank] >= 10
            
            if threeHigh and pairHigh:
                return (1, "highHigh")
            elif threeHigh or pairHigh:
                return (1, "highLow")
            else:
                return (1, "lowLow")
        
        # calculate how many cards we need to make a full house
        if len(sortedRanks) >= 2:
            if sortedRanks[0][1] >= 3:
                # We have three of a kind, need another pair
                missingForPair = 2 - sortedRanks[1][1]
                threeHigh = self.valueDict[sortedRanks[0][0]] >= 10
                pairHigh = self.valueDict[sortedRanks[1][0]] >= 10
                
                if threeHigh and pairHigh:
                    return (-missingForPair, "highHigh")
                elif threeHigh or pairHigh:
                    return (-missingForPair, "highLow")
                else:
                    return (-missingForPair, "lowLow")
            elif sortedRanks[0][1] == 2 and sortedRanks[1][1] == 2:
                # We have two pairs, need one more card for three of a kind
                firstHigh = self.valueDict[sortedRanks[0][0]] >= 10
                secondHigh = self.valueDict[sortedRanks[1][0]] >= 10
                
                if firstHigh and secondHigh:
                    return (-1, "highHigh")
                elif firstHigh or secondHigh:
                    return (-1, "highLow")
                else:
                    return (-1, "lowLow")
            elif sortedRanks[0][1] == 2:
                # We have a pair, need one more for the pair and one more for three of a kind
                pairHigh = self.valueDict[sortedRanks[0][0]] >= 10
                
                if pairHigh:
                    return (-3, "highLow")
                else:
                    return (-3, "lowLow")
        elif len(sortedRanks) == 1:
            if sortedRanks[0][1] == 2:
                # We have one pair, need three more cards
                pairHigh = self.valueDict[sortedRanks[0][0]] >= 10
                
                if pairHigh:
                    return (-3, "highLow")
                else:
                    return (-3, "lowLow")
            elif sortedRanks[0][1] == 1:
                # We have one card of a kind, need four more cards
                cardHigh = self.valueDict[sortedRanks[0][0]] >= 10
                
                if cardHigh:
                    return (-4, "highLow")
                else:
                    return (-4, "lowLow")
        
        return (-5, None)  # worst case: need 5 cards to make a full house

    # 1 if we have a straight flush
    # negative number representing how many cards away we are from making a straight flush
    def have_straight_flush(self, cards):
        if len(cards) < 5:
            return (-5, None)  # we need at least 5 cards to make a straight flush
        
        # check each suit
        suits = ["H", "D", "C", "S"]
        bestDistance = -5  # worst case
        bestQuality = None
        
        for suit in suits:
            # get all cards of this suit
            suitedCards = [card for card in cards if card[0] == suit]
            
            if len(suitedCards) >= 5:
                # We have enough cards of this suit for a straight flush
                # Check if we have a straight with these cards
                straightResult = self.have_straight(suitedCards)
                if straightResult[0] == 1:
                    return (1, straightResult[1])  # we have a straight flush with the same quality
                elif straightResult[0] > bestDistance:
                    bestDistance = straightResult[0]
                    bestQuality = straightResult[1]
            else:
                # Not enough cards of this suit
                suitDistance = -(5 - len(suitedCards))
                if suitDistance > bestDistance:
                    bestDistance = suitDistance
                    # Quality is undetermined if we don't have enough cards
                    bestQuality = None
        
        return (bestDistance, bestQuality)

    # 1 if we have a royal flush
    # negative number representing how many cards away we are from making a royal flush
    def have_royal_flush(self, cards):
        if len(cards) < 5:
            return (-5, None)  # we need at least 5 cards to make a royal flush
        
        # check each suit
        suits = ["H", "D", "C", "S"]
        bestDistance = -5  # worst case
        
        for suit in suits:
            royalRanks = ["T", "J", "Q", "K", "A"]
            # Check which royal cards we have for this suit
            missingCards = 0
            for rank in royalRanks:
                if not any(card[0] == suit and card[1] == rank for card in cards):
                    missingCards += 1
            
            if missingCards == 0:
                return (1, "high")  # Royal flush is always high
            
            distance = -missingCards
            if distance > bestDistance:
                bestDistance = distance
        
        return (bestDistance, "high")  # Royal flush is always categorized as high


    def get_abstract_state(self, holeCards, communityCards, street):
        allCards = holeCards + communityCards

        # handle preflop separately
        if street == "preflop":
            return self.pre_flop_abstraction(holeCards)

        # Dictionary to track the best hand type with community cards only
        bestCommunityHand = {
            "type": "highCard",  # default is high card
            "value": 0,          # to compare same-type hands
            "quality": None      # high/low designation
        }
        
        # Dictionary to track the best hand with all cards
        bestAllHand = {
            "type": "highCard",  # default is high card
            "value": 0,          # to compare same-type hands
            "quality": None      # high/low designation
        }
        
        # calculate what we have with just community cards
        pairComm = self.have_of_a_kind(communityCards, 2)
        threeComm = self.have_of_a_kind(communityCards, 3)
        fourComm = self.have_of_a_kind(communityCards, 4)
        twoPairComm = self.have_two_pair(communityCards)
        straightComm = self.have_straight(communityCards)
        flushComm = self.have_flush(communityCards)
        fullHouseComm = self.have_full_house(communityCards)
        straightFlushComm = self.have_straight_flush(communityCards)
        royalFlushComm = self.have_royal_flush(communityCards)

        # calculate what we have with all cards
        pair = self.have_of_a_kind(allCards, 2)
        three = self.have_of_a_kind(allCards, 3)
        four = self.have_of_a_kind(allCards, 4)
        twoPair = self.have_two_pair(allCards)
        straight = self.have_straight(allCards)
        flush = self.have_flush(allCards)
        fullHouse = self.have_full_house(allCards)
        straightFlush = self.have_straight_flush(allCards)
        royalFlush = self.have_royal_flush(allCards)

        # Determine highest card in hole cards and community cards
        highestHoleCard = max([self.valueDict[card[1]] for card in holeCards], default=0)
        highestCommCard = max([self.valueDict[card[1]] for card in communityCards], default=0)

        # Define hand ranking from lowest to highest
        handRanking = [
            "highCard", "pair", "twoPair", "three", "straight", 
            "flush", "fullHouse", "four", "straightFlush", "royalFlush"
        ]
        
        # Determine best hand with community cards only
        if royalFlushComm[0] == 1:
            bestCommunityHand = {"type": "royalFlush", "value": 14, "quality": "high"}
        elif straightFlushComm[0] == 1:
            bestCommunityHand = {"type": "straightFlush", "value": 0, "quality": straightFlushComm[1]} # Value will need to be set based on highest card
        elif fourComm[0] == 1:
            bestCommunityHand = {"type": "four", "value": 0, "quality": fourComm[1]} # Value will need to be the rank of the four of a kind
        elif fullHouseComm[0] == 1:
            bestCommunityHand = {"type": "fullHouse", "value": 0, "quality": fullHouseComm[1]} # Value will be the rank of the three of a kind
        elif flushComm[0] == 1:
            bestCommunityHand = {"type": "flush", "value": highestCommCard, "quality": flushComm[1]}
        elif straightComm[0] == 1:
            bestCommunityHand = {"type": "straight", "value": 0, "quality": straightComm[1]} # Value will be the highest card in the straight
        elif threeComm[0] == 1:
            bestCommunityHand = {"type": "three", "value": 0, "quality": threeComm[1]} # Value will be the rank of the three of a kind
        elif twoPairComm[0] == 1:
            bestCommunityHand = {"type": "twoPair", "value": 0, "quality": twoPairComm[1]} # Value will be the higher pair
        elif pairComm[0] == 1:
            bestCommunityHand = {"type": "pair", "value": 0, "quality": pairComm[1]} # Value will be the rank of the pair
        else:
            bestCommunityHand = {"type": "highCard", "value": highestCommCard, "quality": "high" if highestCommCard >= 10 else "low"}
        
        # Determine best hand with all cards
        if royalFlush[0] == 1:
            bestAllHand = {"type": "royalFlush", "value": 14, "quality": "high"}
        elif straightFlush[0] == 1:
            bestAllHand = {"type": "straightFlush", "value": 0, "quality": straightFlush[1]}
        elif four[0] == 1:
            bestAllHand = {"type": "four", "value": 0, "quality": four[1]}
        elif fullHouse[0] == 1:
            bestAllHand = {"type": "fullHouse", "value": 0, "quality": fullHouse[1]}
        elif flush[0] == 1:
            # For flush, we need to determine if the highest card in the flush is from hole or community cards
            bestAllHand = {"type": "flush", "value": max(highestHoleCard, highestCommCard), "quality": flush[1]}
        elif straight[0] == 1:
            bestAllHand = {"type": "straight", "value": 0, "quality": straight[1]}
        elif three[0] == 1:
            bestAllHand = {"type": "three", "value": 0, "quality": three[1]}
        elif twoPair[0] == 1:
            bestAllHand = {"type": "twoPair", "value": 0, "quality": twoPair[1]}
        elif pair[0] == 1:
            bestAllHand = {"type": "pair", "value": 0, "quality": pair[1]}
        else:
            bestAllHand = {"type": "highCard", "value": max(highestHoleCard, highestCommCard), 
                        "quality": "high" if max(highestHoleCard, highestCommCard) >= 10 else "low"}
        
        # Initialize validBuckets with updated street-specific drawing hand buckets
        validBuckets = {
            # Community best flags
            "communityBestF": False,
            "communityBestT": False,
            "communityBestR": False,
            
            # Made hands
            "highCardLow": False, "highCardHigh": False,
            "pairLow": False, "pairHigh": False,
            "twoPairLowLow": False, "twoPairHighLow": False, "twoPairHighHigh": False,
            "threeLow": False, "threeHigh": False,
            "straightLow": False, "straightHigh": False,
            "flush": False,
            "fullHouseLowLow": False, "fullHouseHighLow": False, "fullHouseHighHigh": False,
            "fourLow": False, "fourHigh": False,
            "straightFlush": False,
            "royalFlush": False,
            
            # Drawing hands on the flop (-1 with 2 cards to come)
            "straightLow-1F": False, "straightHigh-1F": False,
            "flush-1F": False, "flush-2": False,
            "straightFlush-1F": False,
            
            # Drawing hands on the turn (-1 with 1 card to come)
            "straightLow-1T": False, "straightHigh-1T": False,
            "flush-1T": False,
            "straightFlush-1T": False
        }
        
        # Compare hand rankings to determine if community cards make the best hand
        commHandRank = handRanking.index(bestCommunityHand["type"])
        allHandRank = handRanking.index(bestAllHand["type"])
        
        # Set communityBest flag based on the current street if the community cards make a better or equal hand
        isCommunityBest = False
        if commHandRank > allHandRank:
            # Community cards make a better hand type
            isCommunityBest = True
        elif commHandRank == allHandRank:
            # Same hand type, need to compare values/qualities
            if bestCommunityHand["value"] >= bestAllHand["value"]:
                # Community cards make a better or equal hand
                isCommunityBest = True
            
            # For high card, check if the highest card is in community
            if bestAllHand["type"] == "highCard" and highestCommCard >= highestHoleCard:
                isCommunityBest = True
        
        # Set the appropriate communityBest flag based on the current street
        if isCommunityBest:
            if street == "flop":
                validBuckets["communityBestF"] = True
            elif street == "turn":
                validBuckets["communityBestT"] = True
            elif street == "river":
                validBuckets["communityBestR"] = True
        
        # Now set the other bucket flags based on the best hand we can make
        if not isCommunityBest:
            # We can make a better hand with our hole cards, so set the corresponding bucket
            
            # highCard
            if bestAllHand["type"] == "highCard":
                if bestAllHand["quality"] == "high":
                    validBuckets["highCardHigh"] = True
                else:
                    validBuckets["highCardLow"] = True
                    
            # pair
            elif bestAllHand["type"] == "pair":
                if bestAllHand["quality"] == "high":
                    validBuckets["pairHigh"] = True
                else:
                    validBuckets["pairLow"] = True
                    
            # twoPair
            elif bestAllHand["type"] == "twoPair":
                if bestAllHand["quality"] == "highHigh":
                    validBuckets["twoPairHighHigh"] = True
                elif bestAllHand["quality"] == "highLow":
                    validBuckets["twoPairHighLow"] = True
                else:
                    validBuckets["twoPairLowLow"] = True
                    
            # three of a kind
            elif bestAllHand["type"] == "three":
                if bestAllHand["quality"] == "high":
                    validBuckets["threeHigh"] = True
                else:
                    validBuckets["threeLow"] = True
                    
            # straight
            elif bestAllHand["type"] == "straight":
                if bestAllHand["quality"] == "high":
                    validBuckets["straightHigh"] = True
                else:
                    validBuckets["straightLow"] = True
                    
            # flush
            elif bestAllHand["type"] == "flush":
                validBuckets["flush"] = True
                
            # fullHouse
            elif bestAllHand["type"] == "fullHouse":
                if bestAllHand["quality"] == "highHigh":
                    validBuckets["fullHouseHighHigh"] = True
                elif bestAllHand["quality"] == "highLow":
                    validBuckets["fullHouseHighLow"] = True
                else:
                    validBuckets["fullHouseLowLow"] = True
                    
            # four
            elif bestAllHand["type"] == "four":
                if bestAllHand["quality"] == "high":
                    validBuckets["fourHigh"] = True
                else:
                    validBuckets["fourLow"] = True
                    
            # straightFlush
            elif bestAllHand["type"] == "straightFlush":
                validBuckets["straightFlush"] = True
                
            # royalFlush
            elif bestAllHand["type"] == "royalFlush":
                validBuckets["royalFlush"] = True
        
        # Check for drawing hands (-1/-2) only if they're better than what community cards offer
        if not isCommunityBest:
            # Helper function to check if hole cards contribute to the hand
            def hole_cards_contribute(hand_type, value=None):
                if hand_type == "straight":
                    # Check if any hole card is part of the potential straight
                    straightCards = []
                    allValues = sorted([self.valueDict[card[1]] for card in allCards])
                    
                    # Find consecutive values
                    for i in range(len(allValues) - 1):
                        if allValues[i] + 1 == allValues[i+1] or allValues[i] == allValues[i+1]:
                            straightCards.append(allValues[i])
                    if len(allValues) > 0:
                        straightCards.append(allValues[-1])
                    
                    return any(self.valueDict[holeCard[1]] in straightCards for holeCard in holeCards)
                    
                elif hand_type == "flush":
                    # Check if any hole card contributes to the potential flush
                    suits_count = {"H": 0, "D": 0, "C": 0, "S": 0}
                    
                    # Count community card suits
                    for card in communityCards:
                        suits_count[card[0]] += 1
                    
                    # Check if any hole card contributes to a potential flush
                    for holeCard in holeCards:
                        if suits_count[holeCard[0]] + 1 >= 3:  # At least 3 cards of the same suit
                            return True
                    return False
                    
                elif hand_type == "straightFlush":
                    # Check if hole cards contribute to both a straight and a flush
                    return hole_cards_contribute("straight") and hole_cards_contribute("flush")
                
                return False

            # Check for almost straight - separate flop (-1F) and turn (-1T) buckets
            if straight[0] == -1 and hole_cards_contribute("straight"):
                if street == "flop":
                    if straight[1] == "high":
                        validBuckets["straightHigh-1F"] = True
                    else:
                        validBuckets["straightLow-1F"] = True
                elif street == "turn":
                    if straight[1] == "high":
                        validBuckets["straightHigh-1T"] = True
                    else:
                        validBuckets["straightLow-1T"] = True
            
            # Check for almost flush - separate flop (-1F) and turn (-1T) buckets
            if flush[0] == -1 and hole_cards_contribute("flush"):
                if street == "flop":
                    validBuckets["flush-1F"] = True
                elif street == "turn":
                    validBuckets["flush-1T"] = True
            elif flush[0] == -2 and street == "flop" and hole_cards_contribute("flush"):
                validBuckets["flush-2"] = True
            
            # Check for almost straight flush - separate flop (-1F) and turn (-1T) buckets
            if straightFlush[0] == -1 and hole_cards_contribute("straightFlush"):
                if street == "flop":
                    validBuckets["straightFlush-1F"] = True
                elif street == "turn":
                    validBuckets["straightFlush-1T"] = True
        
        # go through the valid buckets and find the most valuable one
        if street == "flop":
            flopValueRanking = [
                "royalFlush", "straightFlush", "fourHigh", "fourLow",
                "fullHouseHighHigh", "fullHouseHighLow", "fullHouseLowLow",
                "flush", "straightHigh", "straightLow", "threeHigh", "threeLow",
                "straightFlush-1F", "flush-1F", "twoPairHighHigh", "twoPairHighLow", "twoPairLowLow",
                "straightHigh-1F", "straightLow-1F", "pairHigh", "pairLow", "flush-2",
                "highCardHigh", "highCardLow", "communityBestF"
            ]
            for i in flopValueRanking:
                if validBuckets[i]:
                    return i
                
        elif street == "turn":
            turnValueRanking = [
                "royalFlush", "straightFlush", "fourHigh", "fourLow",
                "fullHouseHighHigh", "fullHouseHighLow", "fullHouseLowLow",
                "flush", "straightHigh", "straightLow", "threeHigh", "threeLow",
                "twoPairHighHigh", "twoPairHighLow", "twoPairLowLow",
                "straightFlush-1T", "flush-1T", "pairHigh", "pairLow",
                "straightHigh-1T", "straightLow-1T", "highCardHigh", "highCardLow", "communityBestT"
            ]
            for i in turnValueRanking:
                if validBuckets[i]:
                    return i
                
        else:  # river
            riverValueRanking = [
                "royalFlush", "straightFlush", "fourHigh", "fourLow",
                "fullHouseHighHigh", "fullHouseHighLow", "fullHouseLowLow",
                "flush", "straightHigh", "straightLow", "threeHigh", "threeLow",
                "twoPairHighHigh", "twoPairHighLow", "twoPairLowLow",
                "pairHigh", "pairLow", "highCardHigh", "highCardLow", "communityBestR"
            ]
            for i in riverValueRanking:
                if validBuckets[i]:
                    return i
        
        return "highCardLow"  # should never be reached, just in case
