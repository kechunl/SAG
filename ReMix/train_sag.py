import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime, pdb, random, pickle
import pandas as pd
import numpy as np
import logging, warnings
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict

from model import abmil_sag as abmil
from tools.utils import setup_logger
from losses.attn_guiding import SAGloss
from losses.build_mtl import get_mtl_opts, build_weighting

warnings.simplefilter('ignore')

def get_bag_feats(feats, bag_label, args):
    if isinstance(feats, str):
        # if feats is a path, load it
        feats = feats.split(',')[0]
        bag_feats = torch.Tensor(np.load(feats)).cuda()
        hg_path = os.path.join(args.hg_dir, os.path.basename(feats))
        if os.path.exists(hg_path):
            hg = torch.Tensor(np.load(hg_path)).cuda()
        elif os.path.exists(hg_path.replace('npy','pickle')):
            with open(hg_path.replace('npy','pickle'), 'rb') as handle:
                hg = pickle.load(handle)
            if np.isnan(hg[5]).any():
                # some slices don't have super melanocytes. attn_map will be nan.
                with open(hg_path.replace('npy','pickle').replace('super_melanocyte_area', 'melanocyte_num'), 'rb') as handle:
                    hg = pickle.load(handle) 
            hg = torch.Tensor(hg[7].reshape(-1)).cuda()
        else:
            hg = torch.zeros(bag_feats.shape[0]).cuda()
        
        tg_path = os.path.join(args.tg_dir, os.path.basename(feats))
        if os.path.exists(tg_path):
            tg = torch.Tensor(np.load(os.path.join(args.tg_dir, os.path.basename(feats)))).cuda()
        elif os.path.exists(tg_path.replace('npy','pickle')):
            with open(tg_path.replace('npy','pickle'), 'rb') as handle:
                tg = pickle.load(handle)
            tg = torch.Tensor(tg[7].reshape(-1)).cuda()
        else:
            tg = torch.zeros(bag_feats.shape[0]).cuda()
    
    shuffle_index = np.random.permutation(len(bag_feats))
    bag_feats = bag_feats[shuffle_index]
    hg = hg[shuffle_index]
    tg = tg[shuffle_index]

    if args.num_classes != 1:
        # mannual one-hot encoding, following dsmil
        label = np.zeros(args.num_classes)
        if int(bag_label) <= (len(label) - 1):
            label[int(bag_label)] = 1
        bag_label = Variable(torch.FloatTensor([label]).cuda())
        
    return bag_label, bag_feats, hg, tg


def inverse_convert_label(labels):
    # one-hot decoding
    if len(np.shape(labels)) == 1:
        return labels
    else:
        converted_labels = np.zeros(len(labels))
        for ix in range(len(labels)):
            converted_labels[ix] = np.argmax(labels[ix])
        return converted_labels


def get_sag_loss(losses, attention, hg, tg, criterion_hg, criterion_tg, args):
    '''
    Inputs:
        - losses: Tensor. length=args.task_num
        - attention: 1 x N, sum=1
        - hg: N, sum=1
        - tg: N, sum=1
        - criterion_hg:
        - criterion_tg:
        - args:
    '''
    if 'hg' in args.sag:
        losses[1] = 0.1 * criterion_hg(attention, hg)
    if 'tg' in args.sag:
        losses[-1] = 0.1 * criterion_tg(attention, tg)
    return losses
    

def train(train_feats, train_labels, milnet, criterion, criterion_hg, criterion_tg, optimizer, mtl, mtl_args, args):
    milnet.train()
    total_loss = 0
    for i in range(len(train_feats)):
        optimizer.zero_grad()
        bag_label, bag_feats, bag_hg, bag_tg = get_bag_feats(train_feats[i], train_labels[i], args)
        # abort invalid features
        if torch.isnan(bag_feats).sum() > 0:
            continue

        bag_prediction, bag_attention = milnet(bag_feats)
        losses = torch.zeros(args.task_num).cuda()
        losses[0] = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        losses = get_sag_loss(losses, bag_attention, bag_hg, bag_tg, criterion_hg, criterion_tg, args)

        # multi task learning
        w = mtl.backward(losses, device=bag_feats.device, **mtl_args)

        optimizer.step()
        bag_loss = torch.mul(losses.cpu().detach(), torch.Tensor(w)).sum()
        total_loss = total_loss + losses[0] # change to only reporting cls loss
        str_losses = ' '.join([f"{number:.{4}f}" for number in losses])
        str_w = ' '.join([f"{number:.{4}f}" for number in w])
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f losses: %s weight: %s' % (i, len(train_feats), bag_loss, str_losses, str_w))
    sys.stdout.write('\n')
    return total_loss / len(train_feats)


def test(test_feats, test_gts, milnet, criterion, args):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        for i in range(len(test_feats)):
            bag_label, bag_feats, _, _ = get_bag_feats(test_feats[i], test_gts[i], args)
            bag_feats = bag_feats.view(-1, args.feats_size)

            bag_prediction, _ = milnet(bag_feats)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            loss = bag_loss

            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_feats), loss.item()))
            test_labels.extend([bag_label.cpu().numpy()])
            test_predictions.extend([(torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
        sys.stdout.write('\n')
    test_labels = np.array(test_labels)
    test_labels = test_labels.reshape(len(test_labels), -1)
    test_predictions = np.array(test_predictions)
    _, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes)
    if args.num_classes == 1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    y_pred, y_true = inverse_convert_label(test_predictions), inverse_convert_label(test_labels)
    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    avg = np.mean([p, r, acc])
    return p, r, acc, avg


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


def main():
    parser = argparse.ArgumentParser(description='Train MIL models with precomputed features')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='Camelyon16', type=str, choices=['Camelyon16','Camelyon16_precompute','melanoma'], help='Dataset folder name')
    # SAG
    parser.add_argument('--sag', default='none', type=str, nargs='+', choices=['none', 'hg', 'tg'], help='Semantic Attention Guidance. heuristic guidance (hg); tissue guidance (tg)')
    parser.add_argument('--hg_loss', default='mse', type=str, choices=['mse', 'in-out'], help='Criterion for HG')
    parser.add_argument('--tg_loss', default='in-out', type=str, choices=['mse', 'in-out'], help='Criterion for TG')
    parser.add_argument('--hg_dir', type=str, help='Path to heuristic guidance')
    parser.add_argument('--tg_dir', type=str, help='Path to tissue guidance.')
    parser.add_argument('--weighting', default='none', type=str, choices=['none', 'EW', 'UW', 'RLW'], help='multi task weighting learning')
    # Utils
    parser.add_argument('--seed', default=1669, type=int, help='random seed for pytorch')
    parser.add_argument('--exp_name', required=True, help='exp_name')
    parser.add_argument('--num_repeats', default=1, type=int, help='Number of repeats')
    # MTL
    parser = get_mtl_opts(parser)

    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.task_num = 1 + len(args.sag)
    
    train_labels_pth = f'datasets/{args.dataset}/processed/train_bag_labels.npy'
    test_labels_pth = f'datasets/{args.dataset}/processed/test_bag_labels.npy'

    train_feats = open(f'datasets/{args.dataset}/processed/train_list.txt', 'r').readlines()
    train_feats = np.array(train_feats)
    test_feats = open(f'datasets/{args.dataset}/processed/test_list.txt', 'r').readlines()
    test_feats = np.array(test_feats)

    # use first_time to avoid duplicated logs
    first_time = True
    for t in range(args.num_repeats):
        ckpt_pth = setup_logger(args, first_time)
        logging.info(f'current args: {args}')

        # prepare model
        milnet = abmil.BClassifier(args.feats_size, args.num_classes).cuda()

        criterion = nn.BCEWithLogitsLoss()
        criterion_hg = SAGloss(args.hg_loss)
        criterion_tg = SAGloss(args.tg_loss)
        mtl_weighting, mtl_kwargs = build_weighting(vars(args), milnet)
        params = [p for p in milnet.parameters() if p.requires_grad] + [v for k,v in mtl_weighting.named_parameters() if 'share_net' not in k and v.requires_grad]
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

        # loading labels
        train_labels, test_labels = np.load(train_labels_pth), np.load(test_labels_pth)
        train_labels, test_labels = torch.Tensor(train_labels).cuda(), torch.Tensor(test_labels).cuda()
        
        for epoch in range(1, args.num_epochs + 1):
            # shuffle data
            shuffled_train_idxs = np.random.permutation(len(train_labels))
            train_feats, train_labels = train_feats[shuffled_train_idxs], train_labels[shuffled_train_idxs]
            train_loss_bag = train(train_feats, train_labels, milnet, criterion, criterion_hg, criterion_tg, optimizer, mtl_weighting, mtl_kwargs, args)
            logging.info('Epoch [%d/%d] train loss: %.4f' % (epoch, args.num_epochs, train_loss_bag))
            scheduler.step()

        precision, recall, accuracy, avg = test(test_feats, test_labels, milnet, criterion, args)
        torch.save(milnet.state_dict(), ckpt_pth)
        logging.info('Final model saved at: ' + ckpt_pth)
        logging.info(f'Precision, Recall, Accuracy, Avg')
        logging.info(f'{precision*100:.2f} {recall*100:.2f} {accuracy*100:.2f} {avg*100:.2f}')
        first_time = False
            

if __name__ == '__main__':
    main()