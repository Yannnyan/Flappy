from model import *
import numpy as np


class FlappyEnvironment():
    def __init__(self,init_bird_count=10, mutate_precentile=0.3) -> None:
        self.init_bird_count = init_bird_count
        self.mutate_precentile = mutate_precentile
        self.currentGeneration = []
        
    def fitFunction(self,xBirdPos, pipePassed):
        """
        xBirdPos = the x position of the bird
        pipePassed = the number of pipes the bird crossed
        """
        pass

    def crossOver(self,birdModel1, birdModel2):
        """
        Performs CrossOver between two birds brains and creates two children.
        birdModel1 = FlappyBrain()
        birdModel2 = FlappyBrain()
        """

    def rateBirds(self):
        """
        Performs rating of the best fit birds to the lowest rated birds, and returns them in order.
        """

    def nextGeneration(self):
        """
        Creates the next generation of birds
        """
        pass
    def initGeneration(self):
        """
        Creates the first generation of birds
        """
        for i in range(self.init_bird_count):
            brain = FlappyBrain()
            brain.randomize_initialization()
            self.currentGeneration.append(brain)









