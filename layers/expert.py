import torch.nn as nn

# 专家模块
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),  
            nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        return self.net(x)  