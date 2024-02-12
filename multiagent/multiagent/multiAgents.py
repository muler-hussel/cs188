# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodList = [manhattanDistance(newPos, pos) for pos in newFood.asList()]
        ghostList = [manhattanDistance(newPos, s.getPosition()) for s in newGhostStates]
        score = successorGameState.getScore()

        if len(foodList) > 0 and len(ghostList) > 0:
            for s in newGhostStates:
                if s.scaredTimer == 0:
                    score = score - 1 / (1 + min(ghostList)) + 1 / (1 + min(foodList))
                else:
                    score += 1 / (1 + min(foodList))
        return score
        
        

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0, 0)[1]
    def value(self, gameState: GameState, agent, depth):
        if agent >= gameState.getNumAgents():
            agent = 0
            depth += 1
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:
            return self.maxValue(gameState, agent, depth)
        else:
            return self.minValue(gameState, agent, depth)
    def maxValue(self, gameState, agent, depth):
        ret = (float("-inf"), None)
        if not gameState.getLegalActions(agent):
            return self.evaluationFunction(gameState)
        for action in gameState.getLegalActions(agent):
            value = self.value(gameState.generateSuccessor(agent, action), agent + 1, depth)
            if type(value) is tuple:
                value = value[0]
            retval = max(ret[0], value)
            if retval != ret[0]:
                ret = (retval, action)
        return ret
    def minValue(self, gameState, agent, depth):
        ret = (float("inf"), None)
        if not gameState.getLegalActions(agent):
            return self.evaluationFunction(gameState)
        for action in gameState.getLegalActions(agent):
            value = self.value(gameState.generateSuccessor(agent, action), agent + 1, depth)
            if type(value) is tuple:
                value = value[0]
            retval = min(ret[0], value)
            if retval != ret[0]:
                ret = (retval, action)
        return ret

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0, 0, float("-inf"), float("inf"))[1]
    
    def value(self, gameState: GameState, agent, depth, alpha, beta):
        if agent >= gameState.getNumAgents():
            agent = 0
            depth += 1
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:
            return self.maxValue(gameState, agent, depth, alpha, beta)
        else:
            return self.minValue(gameState, agent, depth, alpha, beta)
    def maxValue(self, gameState, agent, depth, alpha, beta):
        ret = (float("-inf"), None)
        if not gameState.getLegalActions(agent):
            return self.evaluationFunction(gameState)
        for action in gameState.getLegalActions(agent):
            value = self.value(gameState.generateSuccessor(agent, action), agent + 1, depth, alpha, beta)
            if type(value) is tuple:
                value = value[0]
            retval = max(ret[0], value)
            if retval != ret[0]:
                ret = (retval, action)
            if ret[0] > beta:
                return ret
            alpha = max(alpha, ret[0])
        return ret
    def minValue(self, gameState, agent, depth, alpha, beta):
        ret = (float("inf"), None)
        if not gameState.getLegalActions(agent):
            return self.evaluationFunction(gameState)
        for action in gameState.getLegalActions(agent):
            value = self.value(gameState.generateSuccessor(agent, action), agent + 1, depth, alpha, beta)
            if type(value) is tuple:
                value = value[0]
            retval = min(ret[0], value)
            if retval != ret[0]:
                ret = (retval, action)
            if ret[0] < alpha:
                return ret
            beta = min(beta, ret[0])
        return ret 
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0, 0)[1]
    
    def value(self, gameState: GameState, agent, depth):
        if agent >= gameState.getNumAgents():
            agent = 0
            depth += 1
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:
            return self.maxValue(gameState, agent, depth)
        else:
            return self.expValue(gameState, agent, depth)
    def maxValue(self, gameState, agent, depth):
        ret = (float("-inf"), None)
        if not gameState.getLegalActions(agent):
            return self.evaluationFunction(gameState)
        for action in gameState.getLegalActions(agent):
            value = self.value(gameState.generateSuccessor(agent, action), agent + 1, depth)
            if type(value) is tuple:
                value = value[0]
            retval = max(ret[0], value)
            if retval != ret[0]:
                ret = (retval, action)
        return ret
    def expValue(self, gameState, agent, depth):
        ret = count = 0
        if not gameState.getLegalActions(agent):
            return self.evaluationFunction(gameState)
        for action in gameState.getLegalActions(agent):
            value = self.value(gameState.generateSuccessor(agent, action), agent + 1, depth)
            if type(value) is tuple:
                value = value[0]
            ret += value
            count += 1
            reta = action
        return (ret, reta)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    score = currentGameState.getScore()

    foodList = [manhattanDistance(newPos, pos) for pos in newFood]

    for s in newGhostStates:
        dis = manhattanDistance(newPos, s.getPosition())
        if s.scaredTimer == 0 and dis > 0:
            score -= 1 / (1 + dis)
        else:
            score += 2 / (1 + dis)
    
    if len(foodList) > 0:
        score += 1 / (1 + min(foodList))
    return score
# Abbreviation
better = betterEvaluationFunction