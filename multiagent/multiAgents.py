# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

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
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    offset = 0

    for ghostState in newGhostStates:
        distToGhost = manhattanDistance(ghostState.getPosition(),newPos)
        if ghostState.scaredTimer > 0:
            if ghostState.scaredTimer >= distToGhost:
                offset += 100/distToGhost
        else:
            if distToGhost==0:
                return 0
            elif distToGhost < 5:
                offset -= 5 / distToGhost

    if currentGameState.getFood().count(False) < newFood.count(False):
        offset += 5

    closestFood = 100

    a = 0

    for x in newFood:
        b=0
        for y in x:
            if y:
                if manhattanDistance(newPos,(a,b)) < closestFood:
                    closestFood = manhattanDistance(newPos,(a,b))
            b += 1
        a += 1

    offset += 5 /closestFood    


    return successorGameState.getScore() + offset

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
    
    


    util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

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
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    -eat closest food as long as ghost will not eat you
    -eat pellet only if ghosts are close-by
        -once pellet is eaten, aggresively chase ghosts as long as in effect
    -
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
                if manhattanDistance(currentPos,(a,b)) < manhattanDistnace(currentPos,closestFood):
                    closestFood = (a,b)
            b += 1
        a += 1

    foodOffset = 1 / manHattanDistance(currentPos,closestFood)

    #setting ghost offsets - basically, causes pacman to run away from dangerous situations
    #as long as pellet is not eaten
    #but if pacman is close to ghosts AND pellet, he will try to eat pellet and ghosts

    minGhostDist = min(ghostManDistances)
    avgGhostDist = float(sum(ghostManDistancse)) / len(ghostManDistances)
    
    if max(scaredTimes) == 0:
        if avgGhostDist < 5 and minGhostDist > 1:
            closestPellet = min([manhattanDistance(capsule,currentPos) for capsule in capsulePos])
            if closestPellet < 3:
                pelletOffset = 5 / closestPellet
                ghostOffset = -1 / minGhostDist
            else:
                pelletOffset = 0
                ghostOffset = -3 / minGhostDist
                

    else:
        closestFeast = sys.maxint #stores the dist of closest eatable ghost
        for i in range(ghostManDistances):
            if ghostManDistances[i] <= scaredTimes[i]:
                if ghostManDistances[i] < closestFeast:
                    closestFeast = ghostManDistances[i]
        eatGhostOffset = 25/closestFeast


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

