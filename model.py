import torch
import torch.nn as nn

class FlappyBrain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        """
        input params:
        y position of bird
        y distance to bottom pipe
        y distance to top pipe
        x distance to the pipe pair
        y velocity of the bird
        """
        self.seq1 = nn.Sequential(
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Linear(3,1)
        )
    
    def forward(self, x):
        return self.seq1(x)
    
    def to_flap(self, out):
        if out > 0.5:
            return True
        else:
            return False

    def randomize_initialization(self):
        self.apply(init_weights)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight,0,1)
        m.bias.data.fill_(0)

    


















