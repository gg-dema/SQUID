import torch.nn as nn
import torch.nn.functional as F
import torch
"""
Source: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
"""

cosine_loss = torch.nn.CosineSimilarity(dim=1)
def great_circle_distance(a, b, dim=1):
    cosine_loss_ = torch.nn.functional.cosine_similarity(a, b, dim)
    cosine = torch.clamp(cosine_loss_, -0.99999, 0.99999)  # clamp necessary to avoid nans
    distance = torch.acos(cosine)
    return distance


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-20

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class ContrastiveLossJointSpace(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """
    def __init__(self, margin):
        super(ContrastiveLossJointSpace, self).__init__()
        self.margin = margin
        self.eps = 1e-20
    def forward(self, output1, output2, target, size_average=True):
        distances = great_circle_distance(output2, output1)
        losses = 0.5 * (
            target.float() * distances +
            (1 - target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)
        )
        return losses.mean() if size_average else losses.sum()

class TripletAngleLoss(nn.Module):
    """
    Triplet Angle loss
    Triplet loss using angle distance as metric
    """

    def __init__(self, margin):
        super(TripletAngleLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = great_circle_distance(anchor, positive)
        distance_negative = great_circle_distance(anchor, negative)
        losses = F.relu(distance_positive + self.margin - distance_negative)
        return losses.mean()



class ContrastiveLossNorm(nn.Module):

    def __init__(self, margin):
        super(ContrastiveLossNorm, self).__init__()
        self.margin = margin
        self.eps = 1e-20

    def forward(self, output, target, size_average=True):
        norm = torch.norm(output, dim=-1)
        norm_distance = (norm - 1).pow(2)
        losses = 0.5 * ((target.T * norm_distance) +
                        (1 + -1 * target) * F.relu(self.margin - (norm_distance + self.eps)).pow(2))
        return losses.mean() if size_average else losses.sum()

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, swap=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.swap = swap

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        if self.swap:
            distance_positive_negative = (positive - negative).pow(2).sum(1)
            distance_negative[distance_positive_negative < distance_negative] = distance_positive_negative[distance_positive_negative < distance_negative]
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

