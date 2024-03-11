import torch
from torch.nn.functional import cross_entropy
#from torch.nn.modules.loss import _WeightedLoss
import numpy
from torch.autograd import Function
import numpy as np
#import keras as K
from sklearn.metrics import confusion_matrix
from medpy import metric


EPSILON = 1e-32


# class LogNLLLoss(_WeightedLoss):
#     __constants__ = ['weight', 'reduction', 'ignore_index']
#
#     def __init__(self, weight=None, size_average=None, reduce=None, reduction=None,
#                  ignore_index=-100):
#         super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
#         self.ignore_index = ignore_index
#
#     def forward(self, y_input, y_target):
#         # y_input = torch.log(y_input + EPSILON)
#         return cross_entropy(y_input, y_target, weight=self.weight,
#                              ignore_index=self.ignore_index)


def classwise_iou(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """
    dims = (0, *range(2, len(output.shape)))
    gt = torch.zeros_like(output).scatter_(1, gt[:, None, :], 1)
    intersection = output*gt
    union = output + gt - intersection
    classwise_iou = (intersection.sum(dim=dims).float() + EPSILON) / (union.sum(dim=dims) + EPSILON)

    return classwise_iou


def classwise_f1(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """

    epsilon = 1e-20
    n_classes = output.shape[1]

    output = torch.argmax(output, dim=1)
    true_positives = torch.tensor([((output == i) * (gt == i)).sum() for i in range(n_classes)]).float()
    selected = torch.tensor([(output == i).sum() for i in range(n_classes)]).float()
    relevant = torch.tensor([(gt == i).sum() for i in range(n_classes)]).float()

    precision = (true_positives + epsilon) / (selected + epsilon)
    recall = (true_positives + epsilon) / (relevant + epsilon)
    classwise_f1 = 2 * (precision * recall) / (precision + recall)

    return classwise_f1


def make_weighted_metric(classwise_metric):
    """
    Args:
        classwise_metric: classwise metric like classwise_IOU or classwise_F1
    """

    def weighted_metric(output, gt, weights=None):

        # dimensions to sum over
        dims = (0, *range(2, len(output.shape)))

        # default weights
        if weights == None:
            weights = torch.ones(output.shape[1]) / output.shape[1]
        else:
            # creating tensor if needed
            if len(weights) != output.shape[1]:
                raise ValueError("The number of weights must match with the number of classes")
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights)
            # normalizing weights
            weights /= torch.sum(weights)

        classwise_scores = classwise_metric(output, gt).cpu()

        return classwise_scores 

    return weighted_metric

jaccard_index = make_weighted_metric(classwise_iou)
f1_score = make_weighted_metric(classwise_f1)

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()



    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def iouFun(outputs: np.array, labels: np.array):

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.mean()


def Evaluation_Metrics(result,GT):
    GT = GT.unsqueeze(1).expand(-1, 2, -1, -1)
    result= result.detach().cpu().numpy()
    GT = GT.detach().cpu().numpy()
    Y=np.reshape(GT,(result.shape[0]*result.shape[2]*result.shape[2],2))
    Y=Y.astype(int)
    P=np.reshape(result,(result.shape[0]*result.shape[2]*result.shape[2],2))
    P=P.astype(int)
    tn, fp, fn, tp=confusion_matrix(Y, P,labels=[0,1]).ravel()
    F1=2*tp/(2*tp+fp+fn)
    iou=tp/(tp+fn+fp)
    Sensitivity=tp/(tp+fn)
    print("IoU  is:  ",iou)
    print("F1_Score is:  ",F1)
    print("Sensitivity  is:  ",Sensitivity)
    return iou,F1,Sensitivity

def precision(predict, target):
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = numpy.atleast_1d(predict.astype(numpy.bool))
    target = numpy.atleast_1d(target.astype(numpy.bool))

    tp = numpy.count_nonzero(predict & target)
    fp = numpy.count_nonzero(predict & ~target)

    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    return precision


#2022-4-19
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)

def batch_pix_accuracy(predict, target, labeled):
    pixel_labeled = labeled.sum()
    pixel_correct = ((predict == target) * labeled).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

def batch_intersection_union(predict, target, num_class, labeled):
    predict = predict * labeled.long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()

def eval_metrics(output, target, num_class):
    _, predict = torch.max(output.data, 1)
    predict = predict + 1
    target = target + 1

    labeled = (target > 0) * (target <= num_class)
    correct, num_labeled = batch_pix_accuracy(predict, target, labeled)
    inter, union = batch_intersection_union(predict, target, num_class, labeled)
    return [np.round(correct, 5), np.round(num_labeled, 5), np.round(inter, 5), np.round(union, 5)]

#2022-4-20
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += val.detach().cpu()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()

# class MetricsAndPrint(object):
#     def __init__(self):
#         super(MetricsAndPrint, self).__init__()

# def _reset_metrics():
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     total_loss = AverageMeter()
#     total_inter, total_union = 0, 0
#     total_correct, total_label = 0, 0
#
# def _update_seg_metrics(correct, labeled, inter, union):
#     total_inter, total_union = 0, 0
#     total_correct, total_label = 0, 0
#     total_correct += correct
#     total_label += labeled
#     total_inter += inter
#     total_union += union
# total_inter, total_union = 0, 0
# total_correct, total_label = 0, 0
# def _get_seg_metrics(correct, labeled, inter, union):
#     total_correct += correct
#     total_label += labeled
#     total_inter += inter
#     total_union += union
#     pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
#     IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
#     mIoU = IoU.mean()
#     return {
#         "Pixel_Accuracy": np.round(pixAcc, 3),
#         "Mean_IoU": np.round(mIoU, 3),
#         "Class_IoU": dict(zip(range(2), np.round(IoU, 3)))
#     }


if __name__ == '__main__':
    output, gt = torch.zeros(3, 2, 5, 5), torch.zeros(3, 5, 5).long()
    print(classwise_iou(output, gt))
