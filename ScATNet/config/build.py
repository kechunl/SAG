from torch import nn
from utilities.print_utilities import *
from torch import optim
import torch
import numpy as np
from utilities.logger import Logger
import torchnet as tnt
import pdb
import os
import torchvision.transforms as transforms
from dataset.transforms import DivideToCrops, DivideToScales, RandomCrop, Normalize, ToTensor, Resize, Zooming, EvalResize
from dataset.transforms import CenterCrop

import LibMTL.weighting as weighting_method

imagenet_nomalization = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def build_breast_transforms(opts):
    #scales = [0.1, 0.25, 0.5, 1.0]
    data_transforms = transforms.Compose([
        Resize(opts['resize1']),
        ToTensor(),
        DivideToCrops(scale_levels=opts['resize2_scale'], crop_size=opts['resize2']),
        imagenet_nomalization
    ])
    return data_transforms

def build_melanoma_transforms(opts):
    data_transforms = {}
    crop_size = max(opts['resize1'])
    msc_transform = DivideToScales if opts['transform'] == 'DivideToScale' else Zooming

    data_transforms['train'] = transforms.Compose(
        [
            Resize(crop_size),
            RandomCrop(size=opts['resize1']),
            msc_transform(scale_levels=opts['resize1_scale'], size=opts['resize1']),
            ToTensor(),
            DivideToCrops(scale_levels=opts['resize2_scale'], crop_size=opts['resize2']),
            imagenet_nomalization
        ]
    )

    data_transforms['valid'] = transforms.Compose([
            EvalResize(opts['resize1'][0]),
            #CenterCrop(size=opts['resize1']),
            msc_transform(scale_levels=opts['resize1_scale'], size=opts['resize1']),
            ToTensor(),
            DivideToCrops(scale_levels=opts['resize2_scale'], crop_size=opts['resize2']),
            imagenet_nomalization])
    data_transforms['test'] = transforms.Compose([
            ToTensor(),
            msc_transform(scale_levels=opts['resize1_scale'], size=opts['resize1']),
            DivideToCrops(scale_levels=opts['resize2_scale'], crop_size=opts['resize2']),
            imagenet_nomalization
        ])
    return data_transforms


def build_criteria(opts):
    '''
    Build the criterion function
    :param opts: arguments
    :return: Loss function
    '''
    if opts['loss_function'] == 'cross_entropy':
        if opts['num_classes'] <= 2:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
    elif opts['loss_function'] == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    else:
        from experiment.label_smoothing import CrossEntropyWithLabelSmoothing
        criterion = CrossEntropyWithLabelSmoothing(smoothing=opts['smoothing'])
    if opts['cuda'] is not None:
        criterion = criterion.to(opts['cuda'])
    return criterion

def build_attn_criteria(opts, head_ind=None):
    '''
    Build the attention guiding criterion function
    :param opts: arguments
    :return: attn loss function
    '''
    from experiment.attn_guiding import AttnGuideReg
    criterion = AttnGuideReg(opts, head_ind=head_ind)
    return criterion

def build_cuda(opts):
    cuda = None
    if opts['use_gpu'] and len(opts['gpu_id']) > 0:
        cuda = 'cuda:' + str(opts['gpu_id'][0])
    opts['cuda'] = cuda
    return opts


def build_class_weights(opts):
    #Counting class distribution in training dataset
    with open(os.path.join(opts['data'], 'train.txt'), 'r') as f:
        image_list = [line.rstrip() for line in f]
    diag_labels = [int(x.split(';')[1]) for x in image_list]
    class_weights = np.histogram(diag_labels, bins=opts['num_classes'])[0]
    class_weights = np.array(class_weights) / np.float32(sum(class_weights))
    for i in range(opts['num_classes']):
        class_weights[i] = round(np.log(1 / class_weights[i]), 5)
    # class_weights = torch.FloatTensor(np.ones(opts['num_classes']))
    opts['class_weights'] = class_weights
    return opts


def build_scheduler(opts, optimizer):
    scheduler = None
    if opts['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts['epochs']//3, gamma=opts['lr_decay'])
    elif opts['scheduler'] == 'cosine':
        from experiment.scheduler import CosineLR
        scheduler = CosineLR(base_lr=opts['lr'], max_epochs=opts['epochs'], optimizer=optimizer)
    elif opts['scheduler'] == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(factor=opts['lr_decay'], patience=opts['patience'],
                                                               min_lr= opts['lr'] / 50, optimizer=optimizer,
                                                               cooldown=5)
    elif opts['scheduler'] == 'cycle':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=opts['lr']/10, max_lr=opts['lr'],
                                                      cycle_momentum=False)

    return scheduler


def build_optimizer(opts, model, weighting):
    '''
    Creates the optimizer
    :param opts: Arguments
    :param model: Medical imaging model.
    :param weighting: MTL weighting module.
    :return: Optimizer
    '''
    optimizer = None
    params = [p for p in model.parameters() if p.requires_grad] + [v for k,v in weighting.named_parameters() if 'share_net' not in k and v.requires_grad]
    if opts['optim'] == 'sgd':
        print_info_message('Using SGD optimizer')
        optimizer = optim.SGD(params, lr=opts['lr'], weight_decay=opts['weight_decay'])
    elif opts['optim'] == 'adam':
        print_info_message('Using ADAM optimizer')
        beta1 = 0.9 if opts['adam_beta1'] is None else opts['adam_beta1']
        beta2 = 0.999 if opts['adam_beta2'] is None else opts['adam_beta2']
        optimizer = optim.Adam(
            params,
            lr=opts['lr'],
            betas=(beta1, beta2),
            weight_decay=opts['weight_decay'],
            eps=1e-9)
    else:
        print_error_message('{} optimizer not yet supported'.format(opts['optim']))
    # sanity check to ensure that everything is fine
    if optimizer is None:
        print_error_message('Optimizer cannot be None. Please check')
    return optimizer


def update_optimizer(optimizer, lr_value):
    '''
    Update the Learning rate in optimizer
    :param optimizer: Optimizer
    :param lr_value: Learning rate value to be used
    :return: Updated Optimizer
    '''
    optimizer.param_groups[0]['lr'] = lr_value
    return optimizer


def read_lr_from_optimzier(optimizer):
    '''
    Utility to read the current LR value of an optimizer
    :param optimizer: Optimizer
    :return: learning rate
    '''
    return optimizer.param_groups[0]['lr']

def build_weighting(opts, model):
    kwargs = {}
    if opts['weighting'] == 'None':
        weighting = weighting_method.__dict__['EW']()
        weighting.share_net = model.module.scale_wise_attn
        weighting.rep_grad = False
        weighting.device = opts['gpu_id'][0]
        weighting.task_num = opts['task_num']
        weighting.init_param()
        return weighting, kwargs
    elif opts['weighting'] in ['EW', 'UW', 'GradNorm', 'GLS', 'RLW', 'MGDA', 'IMTL',
                            'PCGrad', 'GradVac', 'CAGrad', 'GradDrop', 'DWA', 
                            'Nash_MTL', 'MoCo', 'Aligned_MTL']:
        if opts['weighting'] in ['DWA']:
            if opts['T'] is not None:
                kwargs['T'] = opts['T']
            else:
                raise ValueError('DWA needs keyword T')
        elif opts['weighting'] in ['GradNorm']:
            if opts['alpha'] is not None:
                kwargs['alpha'] = opts['alpha']
            else:
                raise ValueError('GradNorm needs keyword alpha')
        elif opts['weighting'] in ['MGDA']:
            if opts['mgda_gn'] is not None:
                if opts['mgda_gn'] in ['none', 'l2', 'loss', 'loss+']:
                    kwargs['mgda_gn'] = opts['mgda_gn']
                else:
                    raise ValueError('No support mgda_gn {} for MGDA'.format(opts['mgda_gn'])) 
            else:
                raise ValueError('MGDA needs keyword mgda_gn')
        elif opts['weighting'] in ['GradVac']:
            if opts['GradVac_beta'] is not None:
                kwargs['GradVac_beta'] = opts['GradVac_beta']
                kwargs['GradVac_group_type'] = opts['GradVac_group_type']
            else:
                raise ValueError('GradVac needs keyword beta')
        elif opts['weighting'] in ['GradDrop']:
            if opts['leak'] is not None:
                kwargs['leak'] = opts['leak']
            else:
                raise ValueError('GradDrop needs keyword leak')
        elif opts['weighting'] in ['CAGrad']:
            if opts['calpha'] is not None and opts['rescale'] is not None:
                kwargs['calpha'] = opts['calpha']
                kwargs['rescale'] = opts['rescale']
            else:
                raise ValueError('CAGrad needs keyword calpha and rescale')
        elif opts['weighting'] in ['Nash_MTL']:
            if opts['update_weights_every'] is not None and opts['optim_niter'] is not None and opts['max_norm'] is not None:
                kwargs['update_weights_every'] = opts['update_weights_every']
                kwargs['optim_niter'] = opts['optim_niter']
                kwargs['max_norm'] = opts['max_norm']
            else:
                raise ValueError('Nash_MTL needs update_weights_every, optim_niter, and max_norm')
        elif opts['weighting'] in ['MoCo']:
            kwargs['MoCo_beta'] = opts['MoCo_beta']
            kwargs['MoCo_beta_sigma'] = opts['MoCo_beta_sigma']
            kwargs['MoCo_gamma'] = opts['MoCo_gamma']
            kwargs['MoCo_gamma_sigma'] = opts['MoCo_gamma_sigma']
            kwargs['MoCo_rho'] = opts['MoCo_rho']
        
        weighting = weighting_method.__dict__[opts['weighting']]()
        weighting.share_net = model.module.scale_wise_attn
        weighting.rep_grad = False
        weighting.device = opts['gpu_id'][0]
        weighting.task_num = opts['task_num']
        weighting.init_param()
        return weighting, kwargs
    else:
        raise ValueError('No support weighting method {}'.format(opts['weighting'])) 

def build_dataset(opts):
    from dataset.dataloaders import create_datasets, create_dataloader
    # -----------------------------------------------------------------------------
    # Preparing Dataset
    # -----------------------------------------------------------------------------
    # data_transforms = build_melanoma_transforms(opts) if opts['dataset'] == 'melanoma' else build_breast_transforms(opts)
    train_set, valid_set, merge_train_valid_set, test_set = create_datasets(opts)
    train_loader, valid_loader, train_valid_loader, test_loader = create_dataloader(train_set, valid_set, merge_train_valid_set, test_set, opts=opts)
    return train_loader, valid_loader, train_valid_loader, test_loader


def build_model(opts):
    seed_everything(opts)
    feature_extractor=None

    if opts['model'] == 'multi_resolution':
        from model.msc_model import MultiScaleAttention as CNN
    elif opts['model'] in ['mobilenetv2', 'mnasnet', 'espnetv2', 'resnet50']:
        from model.baseline_models import baselineModels as CNN
    elif opts['model'] == 'learned_weight':
        from model.baselines.baselines import LearnedWeight as CNN
    elif opts['model'] == 'mean_weight':
        from model.baselines.baselines import MeanWeight as CNN
    elif opts['model'] == 'self_attn':
        from model.baselines.baselines import SelfAttn as CNN
    else:
        print_error_message("unsupported model")

    from model.base_feature_extractor import BaseFeatureExtractor
    model = CNN(opts)
    if opts['base_extractor'] != None:
        feature_extractor = BaseFeatureExtractor(opts)
        feature_extractor.eval()
    if opts['resume'] is not None and opts['resume'] != '':
        saved_dict = torch.load(opts['resume'])
    if opts['use_gpu'] and len(opts['gpu_id']) > 0:
        if opts['use_parallel']:
            model = nn.DataParallel(model, device_ids=opts['gpu_id'])
            if feature_extractor is not None:
                print_info_message('Feature Extractor on Data Parallel')
                feature_extractor = nn.DataParallel(feature_extractor, device_ids=opts['gpu_id'])
        if opts['model'] != 'kmeans':
            model.to(opts['cuda'])
        if feature_extractor is not None:
            feature_extractor.to(opts['cuda'])
            #print(feature_extractor is None)
        if opts['resume'] is not None and opts['resume'] != '':
            print_info_message('Loaded Model')
            model.load_state_dict(saved_dict)
        if len(opts['gpu_id']) == 1 and opts['model'] != 'kmeans':
            model = model.to(opts['cuda'])

    if opts['resume'] is not None and (opts['cuda'] is None or not opts['use_parallel']):
        print_info_message('Loaded Model')
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in saved_dict.items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v
        #     # load params
        # saved_dict = new_state_dict
    if opts['resume'] is not None and opts['resume'] != '':
        model.load_state_dict(saved_dict)
    if opts['finetune']:
        print_info_message('Freezing batch normalization layers in model')
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    if opts['finetune_base_extractor']:
        feature_extractor.train()
        print_info_message('Freezing batch normalization layers in base extractor')
        for m in feature_extractor.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    return model, feature_extractor

def build_visualization(opts):
    if opts['visdom']:
        print('setting up config...')
        opts['logger'] = Logger(opts)
        opts['confusion_meter'] = tnt.meter.ConfusionMeter(opts['num_classes'], normalized=True)
    else:
        opts['logger'] = None
    return opts


def seed_everything(opts):
    seed = opts['seed']
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
