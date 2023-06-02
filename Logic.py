from birdPool import *
import math
from GeneticAlgorithm import *
from utils import closestPipeInd
import torch

class Logic():
    """
    Contains all the logic for the information passing between modules
    """
    def __init__(self, bird_count, xBird) -> None:
        self.environment = FlappyEnvironment(bird_count, 0.2)
        self.environment.initGeneration()
        self.device = torch.device('cpu')
        self.xBird = xBird
    
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
        closest_pipe = closestPipeInd(upperPipes,self.xBird)
        for i in range(self.environment.init_bird_count):
            brain = self.environment.currentGeneration[i] # the model of the bird
            brain = brain.to(self.device)
            bird = birdPool.birds[i] # bird parameters
            try:
                inp = self.getInputTensor(upperPipes,lowerPipes,closest_pipe,bird)
                with torch.no_grad():
                    events.append(brain.to_flap(brain.forward(inp)))
            except:
                events.append(False)
        return events


    def nextGeneration(self):
        """
        runs the best fit algorithm for the current generation and inits the next generation
        """
        if self.environment.gen_num != 0:
            self.environment.nextGeneration()
        self.environment.gen_num += 1
        return BirdPool(self.environment.init_bird_count)

    def reportCrash(self, info):
        """
        Reports if the bird has crashed
        """
        id = info['id']
        score = info['score']
        groundCrash = info['groundCrash']
        rot = info['playerRot']
        upperPipes = info['upperPipes']
        lowerPipes = info['lowerPipes']
        velY = info['playerVelY']
        y = info['y']
        survive_time = info['survive_time']
        self.environment.fitFunction(score,groundCrash,rot,upperPipes,lowerPipes,velY,y,survive_time, id)










