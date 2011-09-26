# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util, sys

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        newPos = successorGameState.getPacmanPosition() # tuple
        newFood = successorGameState.getFood() # Grid, access with boolean notation or asList()
        newGhostStates = successorGameState.getGhostStates() # AgentState, in ghostAgents.py
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] # List of times

        if newFood.count() == 0:
            return sys.maxint

        foods = newFood.asList()
        foodDists = [abs(newPos[0] - foodPos[0]) + abs(newPos[1] - foodPos[1]) for foodPos in foods]
        minFoodDist = min(foodDists)
        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates if ghostState.scaredTimer == 0]
        ghostDist = [abs(ghostPos[0] - newPos[0]) + abs(ghostPos[1] - newPos[1]) for ghostPos in ghostPositions]
        if len(ghostDist) > 0 and min(ghostDist) <= 1:
            return -sys.maxint
        else:
            return - minFoodDist - (newFood.width * newFood.height * newFood.count())

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        return self.minmax(gameState, 1, 0, max, 'dir')
            
    def minmax(self, gameState, currentDepth, agentIndex, func, returnType):
        # trivial end
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()
          
        # reached end of expansion
        if currentDepth > self.depth:
            return self.evaluationFunction(gameState)
        
        # minmax algorithm
        legalMoves = gameState.getLegalActions(agentIndex)
        nextIndex = agentIndex + 1
        nextDepth = currentDepth
        if nextIndex == gameState.getNumAgents():
            nextIndex = 0
            nextDepth = currentDepth + 1
        if nextIndex == 0:
            nextFunc = max
        else:
            nextFunc = min
        moveResults = [self.minmax(gameState.generateSuccessor(agentIndex, action), nextDepth, nextIndex, nextFunc, 'val') for action in legalMoves]
        desiredResult = func(moveResults)
        if returnType == 'val':
            return desiredResult
        else: # returnType == 'dir'
            bestIndices = [index for index in range(len(moveResults)) if moveResults[index] == desiredResult]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
            #print desiredResult
            return legalMoves[chosenIndex]
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.alphabeta(gameState, 1, 0, max, -sys.maxint, sys.maxint, 'dir')
    
    def alphabeta(self, gameState, currentDepth, agentIndex, func, alpha, beta, returnType):
        # trivial end
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()
          
        # reached end of expansion
        if currentDepth > self.depth:
            return self.evaluationFunction(gameState)
        
        # minmax algorithm with a for loop
        legalMoves = gameState.getLegalActions(agentIndex)
        nextIndex = agentIndex + 1
        nextDepth = currentDepth
        if nextIndex == gameState.getNumAgents():
            nextIndex = 0
            nextDepth = currentDepth + 1
        if nextIndex == 0:
            nextFunc = max
        else:
            nextFunc = min
        
        if func == max:
            bestResult = -sys.maxint
        else:
            bestResult = sys.maxint
            
        for action in legalMoves:
            nextResult = self.alphabeta(gameState.generateSuccessor(agentIndex, action), nextDepth, nextIndex, nextFunc, alpha, beta, 'val')
            if func == max:
                if nextResult >= beta:
                    return nextResult
                elif nextResult > bestResult:
                    bestResult = nextResult
                    bestAction = action
                    
                if nextResult > alpha:
                    alpha = nextResult
                    
            else: # func == min
                if nextResult <= alpha:
                    return nextResult
                elif nextResult < bestResult:
                    bestResult = nextResult
                    bestAction = action
                
                if nextResult < beta:
                    beta = nextResult
             
        if returnType == 'val':
            return bestResult
        else: # returnType == 'dir'
            # print bestResult
            return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.expectimax(gameState, 1, 0, max, 'dir')
    
    def expectimax(self, gameState, currentDepth, agentIndex, func, returnType):
        # trivial end
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()
          
        # reached end of expansion
        if currentDepth > self.depth:
            return self.evaluationFunction(gameState)
        
        # expectimax algorithm
        legalMoves = gameState.getLegalActions(agentIndex)
        nextIndex = agentIndex + 1
        nextDepth = currentDepth
        if nextIndex == gameState.getNumAgents():
            nextIndex = 0
            nextDepth = currentDepth + 1
        if nextIndex == 0:
            nextFunc = max
        else:
            nextFunc = self.average
        moveResults = [self.expectimax(gameState.generateSuccessor(agentIndex, action), nextDepth, nextIndex, nextFunc, 'val') for action in legalMoves]
        desiredResult = func(moveResults)
        if returnType == 'val':
            return desiredResult
        else: # returnType == 'dir'
            # SHOULD ONLY ENTER THIS BLOCK ON A MAX
            bestIndices = [index for index in range(len(moveResults)) if moveResults[index] == desiredResult]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
            #print desiredResult
            return legalMoves[chosenIndex]

    def average(self, values):
     return sum(values, 0.0) / len(values)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:

    Our evaluation function utilizes several offsets to the score of a given state.
    These include the food, ghost, eatGhost, and pellet offsets. All of these offsets
    are added together with the current score of the given state and returned.

    The food offset calculates the Manhattan distance to the nearest food. We then use
    multipliers to provide a reasonable offset (the multipliers depend on the positions of
    ghosts)
    
    The calculation of the ghost and pellet offsets are done together. The basic idea was
    to have Pacman "run away" from dangerous situations (that is, if one or more ghosts are
    very close by Manhattan distance). This would be represented by the ghost offset, which
    is always negative. However, if there is a pellet within a reasonable distance
    while ghosts are close to Pacman, Pacman will try to go for the pellet in order to
    scare and feast on the ghosts. The following are cases that we considered:

        WHEN NO GHOSTS ARE SCARED:
 
          **
            this 1st part checks if ghosts are nearby on average but not TOO close.
          **
            if (Average of distances to ghost < A) AND (Distance to closest ghost > B)
                if (there is a pellet close-by)
                    set a significant pellet offset
                    set a less significant ghost offset depending on distance of closest ghost
                if (no pellet close-by)
                    no pellet offset
                    set a more significant ghost offset depending on distance of closest ghost


          **
            this 2nd part checks if there is a ghost that is VERY close
          **
            if (distance to closest ghost < B)
                set a very significant ghost offset


          **
            this 3rd part checks if ghosts are on average far away while there is no individual ghost nearby
            at the same time, it also checks to see that the ghosts are generally close to one another.
            the reasoning behind this check is to make sure that Pacman isn't in the middle of a bunch of ghosts
            (eg. in the middle of a square of ghosts) and end up being stuck with no exit. by making sure the ghosts
            are far away from Pacman AND close to one another, it should decrease the chance of Pacman getting stuck
          **
            if (average ghost distance > A) AND (distance to closest ghost > C) AND (average distances between ghosts > D)
                no ghost offset
                food offset increases


          **
            this else clause is for every other case when ghosts on average are far away, there is no individual ghost
            that is VERY close, and the average distances between ghosts are fairly large (which can increase Pacman's chance
            of getting stuck)
          **
            else
                set a somewhat significant ghost offset

    The eatGhost offset only comes into play if there are ghosts that are scared. Moreover,
    the scare times of said ghosts must be greater than or equal to the Manhattan distances to
    these ghosts. If Pacman has a reasonable chance to catch a scared ghost, it will aggresively
    go for them. Otherwise, other offsets will have a bigger hand in Pacman's actions.
    
    """

    numAgents = currentGameState.getNumAgents()

    pacmanState = currentGameState.getPacmanState()
    currentPos = currentGameState.getPacmanPosition()

    ghostStates = currentGameState.getGhostStates()
    ghostPositions = currentGameState.getGhostPositions()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    foodPos = currentGameState.getFood()

    capsulePos = currentGameState.getCapsules()
    
    #manhattan distances of pacman pos to ghost positions
    ghostManDistances = [manhattanDistance(currentPos,ghostPos) for ghostPos in ghostPositions]
    
    #offsets are individual "scores" given based on specific features
    #there are 2 ghost offsets - closest only deals with the closest ghost
    #avg deals with average distance to ghosts (which can be useful for deciding when to hunt)
    foodOffset = 0
    ghostOffset = 0
    eatGhostOffset = 0
    pelletOffset = 0
   
    #if the ghost and pacman are on the same spot
    #return -inf; makes sure pacman would never be in a state
    #of which it would die
    if 0 in ghostManDistances:
        return -sys.maxint

    #setting food offset (the reciprocal is used)
    closestFood = (sys.maxint, sys.maxint)

    a = 0

    for x in foodPos:
        b=0
        for y in x:
            if y:
                if manhattanDistance(currentPos,(a,b)) < manhattanDistance(currentPos,closestFood):
                    closestFood = (a,b)
            b += 1
        a += 1

    foodOffset = 3.0 / manhattanDistance(currentPos,closestFood)

    #setting ghost offset - basically, causes pacman to run away from dangerous situations
    #as long as pellet is not eaten
    #but if pacman is close to ghosts AND pellet, he will try to eat pellet and ghosts

    minGhostDist = min(ghostManDistances)
    avgGhostDist = float(sum(ghostManDistances)) / len(ghostManDistances)

    sumOfGhostDistToEachOther = 0
    numCombo = 0

    for i in range(len(ghostPositions)):
        for h in range(len(ghostPositions[i:])):
            sumOfGhostDistToEachOther += manhattanDistance(ghostPositions[i],ghostPositions[h])
            numCombo += 1

    avgGhostSeparation = float(sumOfGhostDistToEachOther) / numCombo
    
    if max(scaredTimes) == 0:
        if avgGhostDist <= 5 and minGhostDist > 1:
            if len(capsulePos) != 0:
                closestPellet = min([manhattanDistance(capsule,currentPos) for capsule in capsulePos])
                if closestPellet < 3:
                    pelletOffset = 10.0 / closestPellet
                    ghostOffset = -1.0 / minGhostDist
            else:
                pelletOffset = 0
                ghostOffset = -7.5 / minGhostDist
        elif minGhostDist <= 1:
            ghostOffset = -10.0 / minGhostDist
        elif avgGhostDist > 5 and minGhostDist > 3 and avgGhostSeparation <= 5:
            foodOffset *= 2.0
        else:
            ghostOffset = -3.0 * minGhostDist
                

    else:
        closestFeast = sys.maxint #stores the dist of closest eatable ghost
        for i in range(len(ghostManDistances)):
            if ghostManDistances[i] <= scaredTimes[i]:
                if ghostManDistances[i] < closestFeast:
                    closestFeast = ghostManDistances[i]
        eatGhostOffset = 25.0/closestFeast


    return foodOffset + ghostOffset + pelletOffset + eatGhostOffset + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

