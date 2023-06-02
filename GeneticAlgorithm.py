from model import *
import numpy as np
from utils import closestPipeInd
from config import XBIRD, SCREENHEIGHT
import numpy as np

GROUND_CRASH_PENALTY = -3
VEL_OPOSITE_PIPE_PENALTY = -3
EXTREME_Y_POSITION_PENALTY = -4
PIPE_PASSED_REWARD = 10



class FlappyEnvironment():
    def __init__(self,init_bird_count=10, mutate_precentile=0.5) -> None:
        self.init_bird_count = init_bird_count
        self.mutate_precentile = mutate_precentile
        self.currentGeneration = []
        self.birdRating = []
        self.gen_num = 0
    

    def fitFunction(self, score:int , groundCrash: bool, rot: int, uperPipes: list, lowerPipes: list, velY: int, y: int,\
                    survive_time, id: int):
        """
        score:int 
        groundCrash: bool,
        rot: int,
        uperPipes: list,
        lowerPipes: list,
        velY: int, 
        y: int
        """
        closest_pipe = closestPipeInd(lowerPipes, XBIRD)
        bird_rate = 0

        # reward positively if bird passed alot of pipes
        bird_rate += score * PIPE_PASSED_REWARD

        # penalize is bird crashes into the ground
        bird_rate += GROUND_CRASH_PENALTY if groundCrash else 0
        
        # penalize if the bird is above the upper pipe and still goes up, or 
        # if the bird is below the lower pipe and stil lgoes down
        # bird_rate += VEL_OPOSITE_PIPE_PENALTY if (velY > 0 and lowerPipes[closest_pipe]['y'] > y) or \
        #                                         (velY < 0 and uperPipes[closest_pipe]['y'] < y) else 0
        
        # penalize if the bird surpasses the screen height (it's assumed that it crashed there)
        bird_rate += EXTREME_Y_POSITION_PENALTY if (y >= SCREENHEIGHT) else 0

        # reward for being close to the middle of the two pipes
        # bird_rate += np.e * np.e ** -np.abs(y - ((lowerPipes[closest_pipe]['y'] + uperPipes[closest_pipe]['y']) / 2))

        # reward for making it alive more often then not
        bird_rate += survive_time / 60

        self.birdRating.append((id, bird_rate))

        

    
    def mutateParam(self, parameter: float):
        """
        performs mutation over a pytorch parameter
        """
        parameter = np.random.uniform(-0.125,0.125)
        return parameter

    def mutate_gen(self, gen: list):
        """
        performs mutation over a list of pytorch parameters
        """
        new_params = []
        for param in gen:
            if np.random.random() < self.mutate_precentile:
                new_params.append(self.mutateParam(param))
            else:
                new_params.append(param)
        return new_params
    
    def getModelsParams(self,model1 : FlappyBrain, model2:FlappyBrain):
        bird1Params = []
        bird2Params = []
        for name,param in model1.named_parameters():
            bird1Params.extend(param.flatten().tolist())
        for name,param in model2.named_parameters():
            bird2Params.extend(param.flatten().tolist())
        return bird1Params, bird2Params

    def crossOver(self,birdModel1: FlappyBrain, birdModel2: FlappyBrain, mutate: bool):
        """
        Performs CrossOver between two birds brains and creates two children.
        birdModel1 = FlappyBrain()
        birdModel2 = FlappyBrain()
        """

        # get parameters of the nn
        params1, params2 = self.getModelsParams(birdModel1,birdModel2)

        # randomly select cut index
        cutoff = np.random.randint(1, len(params1))

        # perform crossover
        childParams1, childParams2 = params1[:cutoff], params2[:cutoff]
        childParams1.extend(params2[cutoff:])
        childParams2.extend(params1[cutoff:])

        # add mutations
        if mutate:
            childParams1 = self.mutate_gen(childParams1)
            childParams2 = self.mutate_gen(childParams2)

        # init models
        child1, child2 = FlappyBrain(), FlappyBrain()

        # assign weights for child 1
        last_i = 0
        for name,param in child1.named_parameters():
            n = param.nelement()
            reshaped_par = torch.FloatTensor(childParams1[last_i:n + last_i],device='cpu').reshape(param.shape)
            param.data = reshaped_par
            last_i = n
        
        # assign weights for child 2
        last_i = 0
        for name,param in child2.named_parameters():
            n = param.nelement()
            reshaped_par = torch.FloatTensor(childParams2[last_i:n + last_i],device='cpu').reshape(param.shape)
            param.data = reshaped_par
            last_i = n

        return child1, child2
    
    def rateBirds(self):
        """
        Performs rating of the best fit birds to the lowest rated birds, and returns them in order.
        """
        return sorted(self.birdRating, key=lambda tup: tup[1], reverse=True)


    def matchParents(self, parent_birds_ratings):
        num_parents = len(parent_birds_ratings)
        num_matchings = int(self.init_bird_count / 2)
        generated_children = []
        need_mutation_ind = np.random.choice(num_matchings,int(num_matchings / 6),False)
        for i in range(num_matchings):
            # perform simple matching of two randomly selected parents
            parent1 = np.random.randint(0,num_parents)
            parent2 = np.random.randint(0,num_parents)
            if i in need_mutation_ind:
                mutate = True
            else:
                mutate = False
            child1, child2 = self.crossOver(self.currentGeneration[parent_birds_ratings[parent1][0]], 
                           self.currentGeneration[parent_birds_ratings[parent2][0]], mutate)
            generated_children.append(child1)
            generated_children.append(child2)
        return generated_children
    

    def nextGeneration(self):
        """
        Creates the next generation of birds
        """
        ratings = self.rateBirds()
        print(ratings)
        best_birds_ratings = ratings[:15]
        # perform crossover on best birds
        next_gen = self.matchParents(best_birds_ratings)
        self.birdRating = [] # lose all previous ratings
        self.currentGeneration = next_gen

    def initGeneration(self):
        """
        Creates the first generation of birds
        """
        for i in range(self.init_bird_count):
            brain = FlappyBrain()
            brain.randomize_initialization()
            self.currentGeneration.append(brain)









