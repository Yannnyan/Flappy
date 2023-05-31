
initialValues = {"playerVelY":  -9,   # player's velocity along Y, default same as playerFlapped
        "playerMaxVelY" :  10,   # max vel along Y, max descend speed
        "playerMinVelY" :  -8,   # min vel along Y, max ascend speed
        "playerAccY" :   1,   # players downward acceleration
        "playerRot" :  45,   # player's rotation
        "playerVelRot" :   3,   # angular speed
        "playerRotThr" :  20,   # rotation threshold
        "playerFlapAcc" :  -9,   # players speed on flapping
        "playerFlapped" : False} # True when player flaps


class Bird():
    def __init__(self, VelY=0, MaxVelY=0, MinVelY=0, AccY=0, Rot=0, VelRot=0, RotThr=0, FlapAcc=0, Flapped=0) -> None:
        self.VelY=VelY
        self.MaxVelY = MaxVelY
        self.MinVelY = MinVelY
        self.AccY = AccY
        self.Rot = Rot
        self.VelRot = VelRot
        self.RotThr = RotThr
        self.FlapAcc = FlapAcc
        self.Flapped = Flapped
        self.x = None
        self.y = None
        self.id = None

    def initBirdProperties(self):
        self.VelY    =  -9   # player's velocity along Y, default same as playerFlapped
        self.MaxVelY =  10   # max vel along Y, max descend speed
        self.MinVelY =  -8   # min vel along Y, max ascend speed
        self.AccY    =   1   # players downward acceleration
        self.Rot     =  45   # player's rotation
        self.VelRot  =   3   # angular speed
        self.RotThr  =  20   # rotation threshold
        self.FlapAcc =  -9   # players speed on flapping
        self.Flapped = False # True when player flaps
    
    def setId(self,id):
        self.id = id


class BirdPool():
    def __init__(self, birdNum) -> None:
        self.birdNum = birdNum
        self.birds = []
        for i in range(self.birdNum):
            bird = Bird()
            bird.initBirdProperties()
            bird.setId(i)
            self.birds.append(bird)


    

    












