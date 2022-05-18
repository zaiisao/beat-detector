import torch
import torch.nn as nn
import math

class FocalLoss(nn.Module):
    def forward(self, classifications, regressions, annotations):
        alpha = 0.25
        gamma = 2.0
        anchor_width = 9
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bline_annotation = annotations[j, :, :]
            #bline_annotation = bline_annotation[bline_annotation[:, 2] != -1] # -1은 padding value
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            beat_indices = torch.nonzero(bline_annotation[0] == 1, as_tuple=False)
            assigned_annotations = torch.cat((
                beat_indices - math.floor(anchor_width/2),
                beat_indices + math.floor(anchor_width/2),
                bline_annotation[1, beat_indices] == 1,
            ), 1).to(annotations.device)

            gt_ctr_x = (assigned_annotations[:, 0] + assigned_annotations[:, 1])/2

            anchor = torch.cat((
                torch.floor(gt_ctr_x.unsqueeze(dim=1) / 10) * 10,
                torch.ceil(gt_ctr_x.unsqueeze(dim=1) / 10 + 0.1) * 10
            ), 1).to(gt_ctr_x.device)

            anchor_widths = anchor[:, 1] - anchor[:, 0]
            anchor_ctr_x = (anchor[:, 0] + anchor[:, 1])/2

            ##########################
            # compute the loss for classification
            # 1280, 2
            targets = torch.zeros(classification.shape)

            if torch.cuda.is_available():
                targets = targets.cuda()

            positive_indices = torch.zeros(classifications.size(dim=1))
            positive_indices[anchor_ctr_x.long()] = 1

            num_positive_anchors = positive_indices.sum()
            targets[gt_ctr_x.long(), assigned_annotations[:, 2].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = (torch.ones(targets.shape) * alpha).cuda()
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            classification_losses.append(cls_loss.sum()/num_positive_anchors)

            ##########################
            # compute the loss for regression

            gt_widths = torch.ones(regression.size(dim=1)).to(regression.device) * 10

            targets_dx = ((gt_ctr_x - anchor_ctr_x) / anchor_widths).cuda()

            regression_diff = torch.abs(targets_dx.long() - regression[positive_indices.long(), :]).float()

            # 9.0 삭제됨. num_box로 추측했고, 명시된 근거가 없음
            regression_loss = torch.where(
                torch.le(regression_diff, 1.0),
                0.5 * torch.pow(regression_diff, 2),
                regression_diff - 0.5
            )
            regression_losses.append(regression_loss.mean())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)
