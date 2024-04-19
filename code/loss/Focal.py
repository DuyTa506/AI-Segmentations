import torch.nn.functional as F

def focal_loss(input, target, weight=None, gamma=2, alpha=None):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    
    # Calculate focal loss
    logpt = F.log_softmax(input, dim=-1)
    pt = torch.exp(logpt)
    logpt = (1 - pt) ** gamma * logpt
    loss = F.nll_loss(logpt, target, weight=weight, reduction="mean", ignore_index=250)
    
    if alpha is not None:
        loss = alpha * loss

    return loss
