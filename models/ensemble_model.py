import torch
import torch.nn as nn


class EnsembleNet(nn.Module):
    def __init__(self):
        super(EnsembleNet, self).__init__()
        self.l1 = nn.Linear(10, 1)
        self.l2 = nn.Linear(10, 1)

    def forward(self, start_logits, end_logits):
        # predicted_logits 10x70
        # 70
        pred_start_logits = self.l1(start_logits)
        pred_end_logits = self.l2(end_logits)
        return pred_start_logits, pred_end_logits
