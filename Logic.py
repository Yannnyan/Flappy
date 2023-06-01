from birdPool import *
import math
from GeneticAlgorithm import *
import torch

def closestPipeInd(pipes, birdPool: BirdPool):
        """
        finds the next pipe to the birds
        """
        xBird = birdPool.birds[0].x
        min_ind = 0
        min_val = math.inf
        for pipe_ind in range(len(pipes)):
            xPipe = pipes[pipe_ind]['x']
            cur_val = abs(xPipe - xBird)
            # checks if the pipe is ahead of the bird and the pipe is the closest
            if  xPipe > xBird and cur_val < min_val:
                min_ind = pipe_ind
                min_val = cur_val
        return min_ind

class Logic():
    """
    Contains all the logic for the information passing between modules
    """
    def __init__(self, bird_count) -> None:
        self.environment = FlappyEnvironment(bird_count)
        self.environment.initGeneration()
        self.device = torch.device('cpu')
    
    def getInputTensor(self, upperPipes, lowerPipes, closest_pipe , bird: Bird):
        yPos = bird.y
        yDistLow = abs(lowerPipes[closest_pipe]['y'] - yPos)
        yDistUp = abs(upperPipes[closest_pipe]['y'] - yPos)
        xDistPair = abs(upperPipes[closest_pipe]['x'] - bird.x)
        yVelo = bird.VelY
        inp = torch.FloatTensor([yPos,yDistLow,yDistUp,xDistPair,yVelo], device=self.device)
        return inp
    

    def passBirdInfo(self, upperPipes, lowerPipes, birdPool: BirdPool):
        """
        passes the input to the model
        upperPipes = {x, y} distances to all upper pipes on screen
        lowerPipes = {x, y} distances to all lower pipes on screen
        """
        events = []
        closest_pipe = closestPipeInd(upperPipes,birdPool)
        for i in range(self.environment.init_bird_count):
            brain = self.environment.currentGeneration[i] # the model of the bird
            brain = brain.to(self.device)
            bird = birdPool.birds[i] # bird parameters
            inp = self.getInputTensor(upperPipes,lowerPipes,closest_pipe,bird)
            with torch.no_grad():
                events.append(brain.to_flap(brain.forward(inp)))
        return events

    def nextGeneration(self):
        """
        runs the best fit algorithm for the current generation and inits the next generation
        """
        self.environment.nextGeneration()

    def reportCrash(self, info):
        """
        Reports if the bird has crashed
        """
        self.environment.fitFunction()








