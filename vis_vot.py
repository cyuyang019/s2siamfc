import os
import glob
import numpy as np
import cv2
import os

def _corner2rect(corners, center=False):
    cx = np.mean(corners[0::2])
    cy = np.mean(corners[1::2])

    x1 = np.min(corners[0::2])
    x2 = np.max(corners[0::2])
    y1 = np.min(corners[1::2])
    y2 = np.max(corners[1::2])

    area1 = np.linalg.norm(corners[0:2] - corners[2:4]) * \
        np.linalg.norm(corners[2:4] - corners[4:6])
    area2 = (x2 - x1) * (y2 - y1)
    scale = np.sqrt(area1 / area2)
    w = scale * (x2 - x1) + 1
    h = scale * (y2 - y1) + 1

    if center:
        return np.array([cx, cy, w, h]).T
    else:
        return np.array([cx - w / 2, cy - h / 2, w, h]).T

seq_name = 'butterfly'

seq_dir = os.path.expanduser('E:/SiamMask/data/VOT2018/'+seq_name+'/')
img_files = sorted(glob.glob(seq_dir + '*.jpg'))

annoes_gt = []
with open(seq_dir + 'groundtruth.txt') as f:
    lines = f.readlines()
    for line in lines:
#    anno = f.readline()
        anno = line.replace(',', ' ').split()
        anno = np.array(anno, dtype=np.float)    
        if len(anno) != 4:
            anno = _corner2rect(anno)
        annoes_gt.append(anno)
annoes_gt = np.array(annoes_gt)

model_path = 'abest_0loss_1.5'
ssiamfc_pre = os.path.expanduser('D:/siamfc-pytorch/results/Comparison/'+model_path+'/baseline/')


# result of ssiamfc
annoes_ssiamfc = []

anno_list = sorted(glob.glob(ssiamfc_pre+seq_name+'/'+seq_name+'_00' + '*.txt'))

for anno in anno_list:
    with open(anno) as f:
        lines = f.readlines()
        for line in lines:
            if len(line) <= 4:
                anno = 0
            else:
                anno = line.replace(',', ' ').split()
                anno = np.array(anno, dtype=np.float)
            annoes_ssiamfc.append(anno)


annoes_siamAN = []

model2_path = 'ablation_self_tracking_BN'
siamAN_pre = os.path.expanduser('D:/siamfc-pytorch/results/Comparison/'+model2_path+'/baseline/')
anno_list = sorted(glob.glob(siamAN_pre+seq_name+'/'+seq_name+'_00' + '*.txt'))

for anno in anno_list:
    with open(anno) as f:
        lines = f.readlines()
        for line in lines:
            if len(line) <= 4:
                anno = 0
            else:
                anno = line.replace(',', ' ').split()
                anno = np.array(anno, dtype=np.float)
            annoes_siamAN.append(anno)

# =============================================================================
# annoes_gt = []
# with open(seq_dir + 'groundtruth.txt') as f:
#     lines = f.readlines()
#     for line in lines:
# #    anno = f.readline()
#         anno = line.replace(',', ' ').split()
#         anno = np.array(anno, dtype=np.float)       
#         anno = _corner2rect(anno)
#         annoes_gt.append(anno)
# annoes_gt = np.array(annoes_gt)
# =============================================================================

thickness = 5
text_sz = 3
text_y = 50
for idx, img_path in enumerate(img_files):
    im = cv2.imread(img_path)
    
    boxes_gt = np.asarray(annoes_gt[idx], dtype=np.int)
    pt1 = (boxes_gt[0], boxes_gt[1])
    pt2 = (boxes_gt[0] + boxes_gt[2], boxes_gt[1] + boxes_gt[3])
    
    im = cv2.rectangle(im, pt1, pt2, (0, 255, 0), thickness=thickness)  #G
    
    if not isinstance(annoes_ssiamfc[idx], int):
        boxes_ssiamfc = np.asarray(annoes_ssiamfc[idx], dtype=np.int)
        pt1 = (boxes_ssiamfc[0], boxes_ssiamfc[1])
        pt2 = (boxes_ssiamfc[0] + boxes_ssiamfc[2], boxes_ssiamfc[1] + boxes_ssiamfc[3])
        im = cv2.rectangle(im, pt1, pt2, (0, 0, 255), thickness=thickness)  #R
    else:
        cv2.putText(im, 'Red Lost', (10, 250+text_y), cv2.FONT_HERSHEY_COMPLEX, text_sz-1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(im, 'Red Lost', (10, 250+text_y), cv2.FONT_HERSHEY_COMPLEX, text_sz-1, (255, 255, 255), 0, cv2.LINE_AA)
        
    if not isinstance(annoes_siamAN[idx], int):
        boxes_siamAN = np.asarray(annoes_siamAN[idx], dtype=np.int)
        pt1 = (boxes_siamAN[0], boxes_siamAN[1])
        pt2 = (boxes_siamAN[0] + boxes_siamAN[2], boxes_siamAN[1] + boxes_siamAN[3])
        im = cv2.rectangle(im, pt1, pt2, (255, 0, 0), thickness=thickness) #B
    else:
        cv2.putText(im, 'Blue Lost', (10, 150+text_y), cv2.FONT_HERSHEY_COMPLEX, text_sz-1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(im, 'Blue Lost', (10, 150+text_y), cv2.FONT_HERSHEY_COMPLEX, text_sz-1, (255, 255,255), 0, cv2.LINE_AA)
    
    cv2.putText(im, '#%d'%idx, (10, 40+text_y), cv2.FONT_HERSHEY_TRIPLEX, text_sz, (0, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('', im)
    if not os.path.exists('./test/'+seq_name+'/'):
        os.mkdir('./test/'+seq_name+'/')
    cv2.imwrite('./test/'+seq_name+'/%d.png'%idx, im)
    cv2.waitKey(60)
    
cv2.destroyAllWindows()
