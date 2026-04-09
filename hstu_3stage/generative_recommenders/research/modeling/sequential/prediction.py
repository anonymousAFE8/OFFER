import torch
import torch.nn as nn


class CEPrediction(nn.Module):
    def __init__(self, dim, class_num):
        super().__init__()
        self.dim = dim
        self.class_num = class_num
        self.w1 = torch.nn.Linear(self.dim, 2 * self.dim, bias=True)
        self.w2 = torch.nn.Linear(2 * self.dim, self.dim, bias=True)
        self.w3 = torch.nn.Linear(self.dim, 2 * self.dim, bias=True)
        self.dropout = torch.nn.Dropout(0.2)
        
        self.prediction_layer = torch.nn.Linear(self.dim, self.class_num, bias=False)

    def forward(self, x): 
        w = self.dropout(self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x)))
        return torch.nn.functional.sigmoid(self.prediction_layer(x + w))
        # return self.prediction_layer(x)
    
class BCEPrediction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.w1 = torch.nn.Linear(self.dim, 2 * self.dim, bias=True)
        self.w2 = torch.nn.Linear(2 * self.dim, self.dim, bias=True)
        self.w3 = torch.nn.Linear(self.dim, 2 * self.dim, bias=True)
        self.dropout = torch.nn.Dropout(0.2)

        self.prediction_bce_layer = torch.nn.Linear(self.dim, 1, bias=False)

    def forward(self, x):
        w = self.dropout(self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x)))
        return torch.nn.functional.sigmoid(self.prediction_bce_layer(x + w))
