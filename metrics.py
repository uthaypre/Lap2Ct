import numpy as np
import cv2
def compute_dice_and_iou(prediction, ground_truth):
    # Flatten arrays to 1D
    prediction = prediction.astype(bool).flatten()
    ground_truth = ground_truth.astype(bool).flatten()

    intersection = np.logical_and(prediction, ground_truth).sum()
    union = np.logical_or(prediction, ground_truth).sum()

    dice = (2. * intersection) / (prediction.sum() + ground_truth.sum() + 1e-8)
    iou = intersection / (union + 1e-8)

    return dice, iou

def compute_dice_iou_multiclass(pred, gt, num_classes):
    dice_scores = {}
    iou_scores = {}

    for class_id in range(num_classes):
        pred_class = (pred == class_id)
        gt_class = (gt == class_id)

        intersection = np.logical_and(pred_class, gt_class).sum()
        union = np.logical_or(pred_class, gt_class).sum()

        dice = (2. * intersection) / (pred_class.sum() + gt_class.sum() + 1e-8)
        iou = intersection / (union + 1e-8)

        dice_scores[class_id] = dice
        iou_scores[class_id] = iou

    return dice_scores, iou_scores

# gt = cv2.imread(r"C:\Users\Pream\OneDrive - ZHAW\Master\Masterarbeit\data\results\image_10624_colored_gt.png", cv2.IMREAD_GRAYSCALE)
# pred = cv2.imread(r"C:\Users\Pream\OneDrive - ZHAW\Master\Masterarbeit\data\results\image_10624_colored_mask.png", cv2.IMREAD_GRAYSCALE)

# intersection = 0
# union = 0
# tp = 0
# tn = 0
# fp = 0
# fn = 0
# for i in range(pred.shape[0]):
#     for j in range(pred.shape[1]):
#         if gt[i,j] == pred[i,j] and gt[i,j] != 0:
#             tp +=1
#         if gt[i,j] == 0 and pred[i,j] == 0:
#             tn +=1
#         if gt[i,j] == 0 and pred[i,j] != 0:
#             fp +=1
#         if gt[i,j] != 0 and pred[i,j] == 0:
#             fn +=1
#         if pred[i,j] != 0:
#             union +=1
#             if gt[i,j] == pred[i,j]:
#                 intersection +=1
#         if gt[i,j] != 0:
#             union +=1

# # dice = intersection*2.0 / (union)



# # print('Dice similarity score is {}'.format(dice))
# print(2*tp / (2*tp + fp + fn))
# print("IoU is: ", tp / (tp + fp + fn))
# print(np.sum(pred!=0))
# print(pred.shape)

# print(intersection)
# print(np.unique(gt))
# print(np.unique(pred))

# dice, iou = compute_dice_iou_multiclass(pred, gt, 2)
# print(np.mean(list(dice.values())))
# print(np.mean(list(iou.values())))

##########################################

import nibabel as nib
import open3d as o3d
import numpy as np

nii_gt = nib.load("/mnt/c/Users/uthaypre/OneDrive - ZHAW/Master/Masterarbeit/data/labelsTs_023/BDMAP_00000005.nii.gz")
gt = nii_gt.get_fdata()
nii_pred = nib.load("/mnt/d/projectsD/datasets/LAP2CT/ct/input_ct_001.nii.gz")
pred = nii_pred.get_fdata()
intersection = 0
union = 0
tp = 0
tn = 0
fp = 0
fn = 0
for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):
        for k in range(pred.shape[2]):
            if gt[i,j,k] == pred[i,j,k] and gt[i,j,k] != 0:
                tp +=1
            if gt[i,j,k] == 0 and pred[i,j,k] == 0:
                tn +=1
            if gt[i,j,k] == 0 and pred[i,j,k] != 0:
                fp +=1
            if gt[i,j,k] != 0 and pred[i,j,k] == 0:
                fn +=1
            if pred[i,j,k] != 0:
                union +=1
                if gt[i,j,k] == pred[i,j,k]:
                    intersection +=1
            if gt[i,j,k] != 0:
                union +=1

print("dice is: ",2*tp / (2*tp + fp + fn))
print("IoU is: ", tp / (tp + fp + fn))
