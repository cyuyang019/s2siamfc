from __future__ import absolute_import

import os
from got10k.experiments import *

#from siamfc.ssiamfc import TrackerSiamFC
#from siamfc.ssiamfc_onlineft import TrackerSiamFC
#from siamfc.siamfc_un_bt import TrackerSiamFC
#from siamfc.siamfc_un_bt_mul import TrackerSiamFC
#from siamfc.siamfc_linear import TrackerSiamFC
from siamfc.siamfc_weight_dropping import TrackerSiamFC


if __name__ == '__main__':
# =============================================================================
#     for b in ['05', '15', '20']:
#         net_path = './checkpoints/b03g%s/siamfc_alexnet_e50.pth'%b
#         tracker = TrackerSiamFC(net_path=net_path, name='hyper_b03_g%s'%b)    
# =============================================================================
#    tracker = TrackerSiamFC(net_path=net_path, name='eccv_best_linear')   
    #ssiam_base
    net_path = './checkpoints/Thesis_rewrite_de/siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path, name='Thesis_rewrite_de')    

# =============================================================================


#         root_dir = 'E:/VID_val_100/Data'
#         VID_exp = ExperimentVID_GOT(root_dir)
#         VID_exp.run(tracker, visualize=False)
#         VID_exp.report([tracker.name])
# =============================================================================

    root_dir = 'E:/SiamMask/data/VOT2016'
    VOT16_exp = ExperimentVOT(root_dir, version=2016, experiments='supervised', read_image=False)
    VOT16_exp.run(tracker, visualize=False)
#        p, ious, fail = VOT16_exp.report([tracker.name])
    
    root_dir = 'E:/SiamMask/data/VOT2018'
    VOT_exp = ExperimentVOT(root_dir, version=2018, experiments='supervised', read_image=False)
    VOT_exp.run(tracker, visualize=False)
#        VOT_exp.report([tracker.name])




    root_dir = 'D:/UDT_pytorch/track/dataset/OTB2015'
    OTB_exp = ExperimentOTB(root_dir, version=2015)
    OTB_exp.run(tracker, visualize=False)
    OTB_exp.report([tracker.name])
    
#    root_dir = 'D:/UDT_pytorch/track/dataset/OTB2013'
#    OTB_exp = ExperimentOTB(root_dir, version=2013)
#    OTB_exp.run(tracker, visualize=False)
#    OTB_exp.report([tracker.name])