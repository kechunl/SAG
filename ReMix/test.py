import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime, pdb, random, pickle, json, tqdm
import pandas as pd
import numpy as np
import logging, warnings
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict

from model import abmil_sag as abmil
from tools.metrics import compute_case
from melanocyte_attention.SAG.ReMix.losses.attn_guiding import SAGloss
from losses.build_mtl import get_mtl_opts, build_weighting

warnings.simplefilter('ignore')

def get_bag_feats(feats, bag_label, args):
    if isinstance(feats, str):
        # if feats is a path, load it
        feats = feats.split(',')[0]
        bag_feats = torch.Tensor(np.load(feats)).cuda()
    
    shuffle_index = np.random.permutation(len(bag_feats))
    bag_feats = bag_feats[shuffle_index]

    if args.num_classes != 1:
        # mannual one-hot encoding, following dsmil
        label = np.zeros(args.num_classes)
        if int(bag_label) <= (len(label) - 1):
            label[int(bag_label)] = 1
        bag_label = Variable(torch.FloatTensor([label]).cuda())
        
    return bag_label, bag_feats


def inverse_convert_label(labels):
    # one-hot decoding
    if len(np.shape(labels)) == 1:
        return labels
    else:
        converted_labels = np.zeros(len(labels))
        for ix in range(len(labels)):
            converted_labels[ix] = np.argmax(labels[ix])
        return converted_labels

def multi_label_roc(labels, predictions, num_classes):
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def test(test_feats, test_gts, milnet, savepath, args):
    milnet.eval()
    test_labels = []
    test_predictions = []
    test_slices = []
    with torch.no_grad():
        for i in range(len(test_feats)):
            bag_label, bag_feats = get_bag_feats(test_feats[i], test_gts[i], args)
            bag_feats = bag_feats.view(-1, args.feats_size)

            bag_prediction, _ = milnet(bag_feats)
            test_slices.append(test_feats[i].split(',')[0])
            test_labels.extend([bag_label.cpu().numpy()])
            test_predictions.extend([(torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])

    test_labels = np.array(test_labels)
    test_labels = test_labels.reshape(len(test_labels), -1)
    test_predictions = np.array(test_predictions)
    aucs, thresholds, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes)
    if args.num_classes == 1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
        y_pred, y_true = inverse_convert_label(test_predictions), inverse_convert_label(test_labels)
        
        p = precision_score(y_true, y_pred, average='macro')
        r = recall_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        auc = np.mean(aucs)
    else:
        scores = test_predictions.copy()
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
        y_pred, y_true = inverse_convert_label(test_predictions), inverse_convert_label(test_labels)
        _, results, auc_macro = compute_case(test_slices, y_pred, y_true, scores, savepath=savepath)
        
        p = results['precision_macro']
        r = results['recall_macro']
        acc = results['overall_accuracy']
        auc = auc_macro
    
    return p, r, acc, auc

def main():
    parser = argparse.ArgumentParser(description='Train MIL models with precomputed features')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--dataset', default='Camelyon16', type=str, choices=['Camelyon16','Camelyon16_precompute', 'melanoma'], help='Dataset folder name')
    # Test
    parser.add_argument('--checkpoint-dir', required=True, type=str, default='results', help='Checkpoint directory location.')
    parser.add_argument('--multiple-dir', action='store_true', default=False)

    args = parser.parse_args()

    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    test_labels_pth = f'datasets/{args.dataset}/processed/test_bag_labels.npy'

    test_feats = open(f'datasets/{args.dataset}/processed/test_list.txt', 'r').readlines()
    test_feats = np.array(test_feats)

    if args.multiple_dir:
        checkpoints = sorted(glob.glob(os.path.join(args.checkpoint_dir, '*.pth')))
    else:
        checkpoints = [args.checkpoint_dir]

    precision_list = []
    recall_list = []
    accuracy_list = []
    auc_list = []
    for checkpoint in tqdm.tqdm(checkpoints, total=len(checkpoints)):
        # prepare model
        milnet = abmil.BClassifier(args.feats_size, args.num_classes).cuda()
        state_dict_weights = torch.load(checkpoint)
        milnet.load_state_dict(state_dict_weights, strict=False)

        # loading labels
        test_labels = np.load(test_labels_pth)
        test_labels = torch.Tensor(test_labels).cuda()
        
        precision, recall, accuracy, auc = test(test_feats, test_labels, milnet, os.path.join(args.checkpoint_dir, 'result_{}'.format(os.path.basename(checkpoint).split('.')[0])), args)
        precision_list.append(precision)
        recall_list.append(recall)
        accuracy_list.append(accuracy)
        auc_list.append(auc)
    results = {'precision': precision_list,
                'recall': recall_list,
                'accuracy': accuracy_list,
                'auc': auc_list,
                'average': [np.mean(precision_list), np.mean(recall_list), np.mean(accuracy_list), np.mean(auc_list)]}
    with open(os.path.join(args.checkpoint_dir, 'results.json'), 'w') as f:
        json.dump(results, f)
            

if __name__ == '__main__':
    main()