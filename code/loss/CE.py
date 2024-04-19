import torch.nn.functional as F
def cross_entropy2d(input, target, weight=None):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, reduction="mean", ignore_index=250
    )
    return loss

def focal_loss(input, target, alpha=0.25, gamma=2, reduction='mean'):
    """
    Focal Loss for multi-class segmentation tasks.
    Args:
        input (torch.Tensor): The input logits (before softmax) from the model, shape (N, C, H, W).
        target (torch.Tensor): The target segmentation mask, shape (N, H, W).
        alpha (float): The weighting factor for class imbalance.
        gamma (float): Focusing parameter to adjust the rate at which easy samples are down-weighted.
        reduction (str): Specifies the reduction to apply to the output. Default is 'mean'.
    Returns:
        torch.Tensor: The computed Focal Loss.
    """
    log_prob = F.log_softmax(input, dim=1)
    prob = torch.exp(log_prob)
    target_one_hot = F.one_hot(target, num_classes=input.size(1)).permute(0, 3, 1, 2).float()

    pt = prob * target_one_hot + (1 - prob) * (1 - target_one_hot)
    focal_weight = (alpha * target_one_hot + (1 - alpha) * (1 - target_one_hot)) * pt.pow(gamma)

    loss = F.binary_cross_entropy_with_logits(input, target_one_hot, reduction='none') * focal_weight
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def dice_loss(input, target, smooth=1):
    """
    Dice Loss for multi-class segmentation tasks.
    Args:
        input (torch.Tensor): The input logits (before softmax) from the model, shape (N, C, H, W).
        target (torch.Tensor): The target segmentation mask, shape (N, H, W).
        smooth (float): Smoothing factor to avoid division by zero.
    Returns:
        torch.Tensor: The computed Dice Loss.
    """
    input = F.softmax(input, dim=1)
    target_one_hot = F.one_hot(target, num_classes=input.size(1)).permute(0, 3, 1, 2).float()

    intersection = torch.sum(input * target_one_hot, dim=(2, 3))
    cardinality = torch.sum(input + target_one_hot, dim=(2, 3))

    dice = (2. * intersection + smooth) / (cardinality + smooth)

    loss = 1 - dice.mean()
    return loss