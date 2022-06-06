import torch
import torch.nn as nn
import math

#   0    0    0    1   0000000
# [00] [00] [00] [01] ...

# beat     0 1 0 1 0 1 0 1 0 1
# downbeat 0 1 0 0 0 0 0 0 0 1

class FocalLoss(nn.Module):
    def __init__(self, sr, target_factor):
        super(FocalLoss, self).__init__()

        self.sr = sr
        self.target_factor = target_factor

    def forward(self, classifications, regressions, annotations):
        alpha = 0.25 # tried: 0.5, 0.01
        gamma = 5 #original 2.0, tried: 1, 5
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        beat_times = annotations[:, 2, :]

        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            # [ [0, 1, 0, 0, 1, ...], [0, 1, 0, 0, 0, ...] ] (beat, downbeat)
            bline_annotation = annotations[j, :, :]
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            beat_indices = torch.nonzero(bline_annotation[0] == 1, as_tuple=False) # [ 3, 16, 20, ... ]
            is_downbeat = bline_annotation[1, beat_indices] == 1 # [ True, False, False, False, True, ... ]

            positive_indices = annotations[j, 0, :]
            num_positive_anchors = positive_indices.sum()

            if num_positive_anchors == 0:
                continue

            ##########################
            # compute the loss for classification
            # 1280, 2

            # 01 beat
            # 10 downbeat
            classification_labels = torch.zeros(classification.shape).to(classification.device)
            for idx, beat_index in enumerate(beat_indices):
                if is_downbeat[idx]:
                    classification_labels[beat_index, 0] = 1
                else:
                    classification_labels[beat_index, 1] = 1

            if torch.cuda.is_available():
                alpha_factor = (torch.ones(classification.shape) * alpha).cuda()
            else:
                alpha_factor = torch.ones(classification.shape) * alpha

            alpha_factor = torch.where(torch.eq(classification_labels, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(classification_labels, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            if -(classification_labels * torch.log(classification)).sum() < 0.001 or -((1.0 - classification_labels) * torch.log(1.0 - classification)).sum() < 0.001:
                print("경고: BCE의 어느 한 쪽은 0.001 미만")

            bce = -(classification_labels * torch.log(classification) + (1.0 - classification_labels) * torch.log(1.0 - classification))

            classification_loss = (focal_weight * bce).sum()/num_positive_anchors * 10

            classification_losses.append(classification_loss)

            ##########################
            # compute the loss for regression

            regression_labels = beat_times[j, :].clone().detach().to(regression.device)
            regression_normal = regression_labels[:] / (regression.shape[0] * self.target_factor / self.sr)

            regression_diff_left = torch.pow(regression_normal[positive_indices.bool()] - regression[positive_indices.bool(), 0], 2).float()

            if positive_indices[-1] == 1:
                positive_indices[-2] = 1
                positive_indices[-1] = 0

            positive_indices_left = positive_indices
            positive_indices_right = torch.roll(positive_indices, 1, 0).to(positive_indices.device)

            regression_diff_left = torch.pow(regression_normal[positive_indices.bool()] - regression[positive_indices_left.bool(), 0], 2).float()
            regression_diff_right = torch.pow(regression_normal[positive_indices.bool()] - regression[positive_indices_right.bool(), 0], 2).float()
            regression_diff = (regression_diff_left + regression_diff_right)/2

            regression_loss = regression_diff.mean() * 10

            regression_losses.append(regression_loss)

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)
