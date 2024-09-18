import os
import numpy as np
import glob
import torch

target_dir = '../datasets/melanoma'
src_dir = '/projects/patho2/melanoma_diagnosis/x10/binarized/49'

case_info = []
with open(os.path.join(src_dir,'train.txt'), 'r') as f:
    case_info += f.readlines()
with open(os.path.join(src_dir,'test.txt'), 'r') as f:
    case_info += f.readlines()
with open(os.path.join(src_dir,'valid.txt'), 'r') as f:
    case_info += f.readlines()


for case in case_info:
    feat_path = case.split(';')[0]
    out_path = os.path.join(target_dir, case.split(';')[1], os.path.basename(case.split(';')[0]).replace('pt','npy'))
    np.save(out_path, np.array(torch.load(feat_path)[2]))
