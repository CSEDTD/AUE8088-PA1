from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('true_positive', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('precision_denom', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('recall_denom', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        cls_num = preds.shape[1]

        preds_label = torch.argmax(preds, dim=1)

        if (preds_label.shape != target.shape):
            raise ValueError
        
        # Find true positive label
        cls = (preds_label == target) # classification result: True or False
        cls_target_pair = torch.stack((cls, target), 1)
        correct_target_pair = cls_target_pair[cls_target_pair[:,0] == True] # filter out False classification
        correct_target = correct_target_pair[:,1]

        self.true_positive = self.true_positive + torch.bincount(correct_target, minlength=cls_num)

        self.precision_denom = self.precision_denom + torch.bincount(preds_label, minlength=cls_num) # preds_label == TP + FP

        self.recall_denom = self.recall_denom + torch.bincount(target, minlength=cls_num) # target == TP + FN

    def compute(self):
        # the labels which self.recall_denom == 0 cannot be true positive, so filter out those labels
        true_positive = self.true_positive[self.recall_denom != 0]
        precision_denom = self.precision_denom[self.recall_denom != 0]
        recall_denom = self.recall_denom[self.recall_denom != 0]

        precision_denom[precision_denom == 0] = 1.0
        recall_denom[recall_denom == 0] = 1.0

        precision = true_positive.float() / precision_denom.float()
        recall = true_positive.float() / recall_denom.float()
        
        f1_score_denom = precision + recall
        f1_score_denom[f1_score_denom == 0] = 1.0

        f1_score = torch.mean((2 * precision * recall) / f1_score_denom)

        return f1_score


class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if (preds.shape != target.shape):
            raise ValueError

        # [TODO] Cound the number of correct prediction
        correct = torch.sum(preds == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
