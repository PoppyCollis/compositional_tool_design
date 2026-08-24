import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class DesignPriorConfig:
    
    # For l1 and l2
    L_MIN = 0.15
    L_MAX = 0.5 # meters; sane band for a link on the franka panda arm
    
    