import torch
from torch.nn.modules.loss import _Loss


class BaselineLoss(_Loss):
    def __init__(self):
        super(BaselineLoss, self).__init__()
        # self.bce = torch.nn.BCELoss()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        target = torch.squeeze(target, dim=1).type(torch.long)
        return self.ce(prediction, target)
