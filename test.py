import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from original_ceshi_LOGO_SAM import medt_net

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os


from utils_LOGO import JointTransform2D, ImageToImage2D, ImageToImage2D_, Image2D
from metrics import jaccard_index, f1_score, dice_coeff, iouFun, Evaluation_Metrics, eval_metrics
import cv2

# tf_val = JointTransform2D(crop=None, p_flip=0, color_jitter_params=None, long_mask=True)
val_dataset_path = 'D:/Data/Code_Dataset/Covid19_text/Test_Folder'
tf_val = JointTransform2D(crop=None, p_flip=0, color_jitter_params=None, long_mask=True)
val_dataset = ImageToImage2D_(val_dataset_path, tf_val)
# predict_dataset = Image2D(val_dataset)
valloader = DataLoader(val_dataset, 1, shuffle=True)
direc = "D:\Data\MedT_SAM\original_LOGO_SAM\MSD"

if __name__ == '__main__':
    device = torch.device("cuda")
    model = medt_net().to(device)
    model.load_state_dict(torch.load("D:/Data/MedT_SAM/original_LOGO_SAM/MSD/_test_output/_weight/best_epoch42_best_iou0.9477_best_local.pth"))
    # model.eval()
    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0
    for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
        if isinstance(rest[0][0], str):
            image_filename = rest[0][0]
        else:
            image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

        X_batch = Variable(X_batch.to(device='cuda'))
        y_batch = Variable(y_batch.to(device='cuda'))
        y_out = model(X_batch)

        # 计算iou指标：
        seg_metrics = eval_metrics(y_out, y_batch, 2)
        total_correct += seg_metrics[0]
        total_label += seg_metrics[1]
        total_inter += seg_metrics[2]
        total_union += seg_metrics[3]
        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIoU = IoU.mean()

        # 保存分割结果图
        tmp = y_out.detach().cpu().numpy()
        tmp[tmp >= 0.5] = 1
        tmp[tmp < 0.5] = 0
        tmp = tmp.astype(int)
        yHaT = tmp

        del X_batch, y_batch, tmp, y_out

        yHaT[yHaT == 1] = 255
        fulldir = direc + "/test_output/"
        if not os.path.isdir(fulldir):
            os.makedirs(fulldir)

        cv2.imwrite(fulldir + image_filename, yHaT[0, 1, :, :])


    print('Acc {:.4f} mIoU {:.4f}'.format(pixAcc, mIoU))