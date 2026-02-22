import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    """
    Edge-based loss function.
    Uses a Laplace operator kernel to extract the edges of the masks,
    and then calculates the BCE loss between the predicted and target edge maps.
    """
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # Define the Laplace operator kernel
        k = torch.Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
        # Register kernel as a buffer since it does not require gradients
        self.register_buffer('kernel', k)
        self.loss_fn = nn.BCELoss()

    def forward(self, pred_mask, target_mask):
        # Ensure the kernel is on the same device as the input tensor
        kernel = self.kernel.to(pred_mask.device)

        # Extract edges using 2D convolution
        pred_edge = F.conv2d(pred_mask, kernel, padding=1)
        target_edge = F.conv2d(target_mask, kernel, padding=1)

        # Constrain edge values to the [0, 1] range
        pred_edge = torch.sigmoid(pred_edge)
        target_edge = torch.sigmoid(target_edge)

        # Calculate the loss between the edge maps
        edge_loss = self.loss_fn(pred_edge, target_edge)
        return edge_loss


class DiceLoss(nn.Module):
    """
    Standard Dice Loss for image segmentation (1 - Dice Coefficient).
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss (1 - IoU).
    """
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        total = (pred + target).sum()
        union = total - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou


class CustomCombinedLoss(nn.Module):
    """
    A weighted combination of standard BCE, Dice, and IoU losses.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.25, iou_weight=0.25, smooth=1e-6):
        super(CustomCombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.bce_criterion = nn.BCELoss()
        self.dice_criterion = DiceLoss(smooth)
        self.iou_criterion = IoULoss(smooth)

    def forward(self, pred, target, bce_weight_map=None):
        """
        Note: 'bce_weight_map' is kept for API compatibility but is no longer used internally.
        """
        bce = self.bce_criterion(pred, target)
        dice = self.dice_criterion(pred, target)
        iou = self.iou_criterion(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice + self.iou_weight * iou


class SegmentationClassificationLoss(nn.Module):
    """
    Joint loss function for simultaneous segmentation and classification.
    Applies standard combined loss for segmentation, BCE for classification,
    and an optional targeted Dice loss specifically for samples with actual defect labels.
    """
    def __init__(self, seg_loss_config, seg_loss_weight=1.0, class_loss_weight=0.5, labeled_dice_weight=0.2,
                 smooth=1e-6):
        super(SegmentationClassificationLoss, self).__init__()
        self.seg_loss_weight = seg_loss_weight
        self.class_loss_weight = class_loss_weight
        self.labeled_dice_weight = labeled_dice_weight

        # Base segmentation loss
        self.seg_criterion = CustomCombinedLoss(
            bce_weight=seg_loss_config['bce_weight'],
            dice_weight=seg_loss_config['dice_weight'],
            iou_weight=seg_loss_config['iou_weight'],
            smooth=smooth
        )
        # Classification loss
        self.class_criterion = nn.BCELoss()

        # Initialize targeted Dice loss if weight > 0
        if self.labeled_dice_weight > 0:
            self.labeled_dice_criterion = DiceLoss(smooth)

    def forward(self, model_output, target_mask, target_class, bce_weight_map=None, dynamic_labeled_dice_weight=None):
        pred_mask, pred_class = model_output

        # 1. Calculate overall segmentation loss
        seg_loss = self.seg_criterion(pred_mask, target_mask)

        # 2. Calculate classification loss
        class_loss = self.class_criterion(pred_class.squeeze(-1), target_class)

        # 3. Combine base losses
        total_loss = self.seg_loss_weight * seg_loss + self.class_loss_weight * class_loss

        # 4. Add targeted Dice loss for labeled samples (samples containing defects)
        current_weight = self.labeled_dice_weight if dynamic_labeled_dice_weight is None else dynamic_labeled_dice_weight

        if current_weight > 0:
            has_label = target_mask.sum(dim=[1, 2, 3]) > 0

            if has_label.any():
                labeled_pred_mask = pred_mask[has_label]
                labeled_target_mask = target_mask[has_label]

                if not hasattr(self, 'labeled_dice_criterion'):
                    self.labeled_dice_criterion = DiceLoss()

                labeled_dice_loss = self.labeled_dice_criterion(labeled_pred_mask, labeled_target_mask)
                total_loss += current_weight * labeled_dice_loss

        return total_loss


class EdgeSegmentationClassificationLoss(nn.Module):
    """
    Extended joint loss function that includes edge-aware penalties.
    Similar to SegmentationClassificationLoss, but adds EdgeLoss for labeled samples
    to improve boundary refinement.
    """
    def __init__(self, seg_loss_config, seg_loss_weight=1.0, class_loss_weight=0.5, labeled_dice_weight=0.2,
                 edge_loss_weight=0.5, smooth=1e-6):
        super(EdgeSegmentationClassificationLoss, self).__init__()
        self.seg_loss_weight = seg_loss_weight
        self.class_loss_weight = class_loss_weight
        self.labeled_dice_weight = labeled_dice_weight
        self.edge_loss_weight = edge_loss_weight

        self.seg_criterion = CustomCombinedLoss(
            bce_weight=seg_loss_config['bce_weight'],
            dice_weight=seg_loss_config['dice_weight'],
            iou_weight=seg_loss_config['iou_weight'],
            smooth=smooth
        )
        self.class_criterion = nn.BCELoss()

        if self.labeled_dice_weight > 0:
            self.labeled_dice_criterion = DiceLoss(smooth)

        if self.edge_loss_weight > 0:
            self.edge_criterion = EdgeLoss()

    def forward(self, model_output, target_mask, target_class, bce_weight_map=None):
        pred_mask, pred_class = model_output

        # 1. Base segmentation loss
        seg_loss = self.seg_criterion(pred_mask, target_mask)

        # 2. Base classification loss
        class_loss = self.class_criterion(pred_class.squeeze(-1), target_class)

        # 3. Combine base losses
        total_loss = self.seg_loss_weight * seg_loss + self.class_loss_weight * class_loss

        # 4. Add targeted Dice and Edge losses for labeled samples
        has_label = target_mask.sum(dim=[1, 2, 3]) > 0

        if has_label.any():
            labeled_pred_mask = pred_mask[has_label]
            labeled_target_mask = target_mask[has_label]

            if self.labeled_dice_weight > 0:
                labeled_dice_loss = self.labeled_dice_criterion(labeled_pred_mask, labeled_target_mask)
                total_loss += self.labeled_dice_weight * labeled_dice_loss

            if self.edge_loss_weight > 0:
                edge_loss = self.edge_criterion(labeled_pred_mask, labeled_target_mask)
                total_loss += self.edge_loss_weight * edge_loss

        return total_loss


class WeightedCombinedLoss(nn.Module):
    """
    A simple weighted combination of BCE, Dice, and IoU losses.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.25, iou_weight=0.25, smooth=1e-6):
        super(WeightedCombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.bce_criterion = nn.BCELoss()
        self.dice_criterion = DiceLoss(smooth)
        self.iou_criterion = IoULoss(smooth)

    def forward(self, pred, target):
        bce = self.bce_criterion(pred, target)
        dice = self.dice_criterion(pred, target)
        iou = self.iou_criterion(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice + self.iou_weight * iou

# ---------------------------------------------------------------------------
# Evaluation Metrics (Numpy Implementations)
# ---------------------------------------------------------------------------

def calculate_dice(mask1, mask2):
    """
    Calculates the Dice Coefficient between two binary masks using Numpy.
    """
    if np.sum(mask1) == 0 and np.sum(mask2) == 0: return 1.0
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2)
    return (2. * intersection) / (union + 1e-8)


def calculate_iou(mask1, mask2):
    """
    Calculates the Intersection over Union (IoU) between two binary masks using Numpy.
    """
    if np.sum(mask1) == 0 and np.sum(mask2) == 0: return 1.0
    intersection = np.sum(mask1 * mask2)
    union = np.sum((mask1 + mask2) > 0)
    return intersection / (union + 1e-8)