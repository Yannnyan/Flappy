import math


def closestPipeInd(pipes, xBird):
        """
        finds the next pipe to the birds
        """
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