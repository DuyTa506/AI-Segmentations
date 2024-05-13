
import sklearn.metrics as skm
import numpy as np
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def intersectionAndUnionGPU(output, target, K, ignore_index=250):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target



'''We have used skelarn libraries to calculate Accuracy and Jaccard Score'''

def get_metrics(gt_label, pred_label):
    #Accuracy Score
    acc = skm.accuracy_score(gt_label, pred_label, normalize=True)
    
    #Jaccard Score/IoU
    js = skm.jaccard_score(gt_label, pred_label, average='micro')
    
    result_gm_sh = [acc, js]
    return(result_gm_sh)

'''
Calculation of confusion matrix from :
https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

Added modifications to calculate 3 evaluation metrics - 
Specificity, Senstivity, F1 Score
'''

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        hist = self.confusion_matrix
        
        TP = np.diag(hist)
        TN = hist.sum() - hist.sum(axis=1) - hist.sum(axis=0) + np.diag(hist)
        FP = hist.sum(axis=1) - TP
        FN = hist.sum(axis=0) - TP
        
        # Calculate Dice Score
        dice_cls = (2 * TP) / (2 * TP + FP + FN + 1e-6)
        dice = np.nanmean(dice_cls)
        
        # Calculate Specificity
        specif_cls = TN / (TN + FP + 1e-6)
        specif = np.nanmean(specif_cls)
        
        # Calculate Sensitivity/Recall
        sensti_cls = TP / (TP + FN + 1e-6)
        sensti = np.nanmean(sensti_cls)
        
        # Calculate Precision
        prec_cls = TP / (TP + FP + 1e-6)
        prec = np.nanmean(prec_cls)
        
        # Calculate F1 Score
        f1 = (2 * prec * sensti) / (prec + sensti + 1e-6)
        
        return {
            "Dice": dice,
            "Specificity": specif,
            "Senstivity": sensti,
            "Precision": prec,
            "F1": f1
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))