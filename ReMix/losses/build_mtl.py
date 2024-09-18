import LibMTL.weighting as weighting_method

def get_mtl_opts(parser):
    'Multi task learning details'
    group = parser.add_argument_group('MTL options')
    ## DWA
    group.add_argument('--T', type=float, default=2.0, help='T for DWA')
    ## MGDA
    group.add_argument('--mgda_gn', default='none', type=str, 
                        help='type of gradient normalization for MGDA, option: l2, none, loss, loss+')
    ## GradVac
    group.add_argument('--GradVac_beta', type=float, default=0.5, help='beta for GradVac')
    group.add_argument('--GradVac_group_type', type=int, default=0, 
                        help='parameter granularity for GradVac (0: whole_model; 1: all_layer; 2: all_matrix)')
    ## GradNorm
    group.add_argument('--alpha', type=float, default=1.5, help='alpha for GradNorm')
    ## GradDrop
    group.add_argument('--leak', type=float, default=0.0, help='leak for GradDrop')
    ## CAGrad
    group.add_argument('--calpha', type=float, default=0.5, help='calpha for CAGrad')
    group.add_argument('--rescale', type=int, default=1, help='rescale for CAGrad')
    ## Nash_MTL
    group.add_argument('--update_weights_every', type=int, default=1, help='update_weights_every for Nash_MTL')
    group.add_argument('--optim_niter', type=int, default=20, help='optim_niter for Nash_MTL')
    group.add_argument('--max_norm', type=float, default=1.0, help='max_norm for Nash_MTL')
    ## MoCo
    group.add_argument('--MoCo_beta', type=float, default=0.5, help='MoCo_beta for MoCo')
    group.add_argument('--MoCo_beta_sigma', type=float, default=0.5, help='MoCo_beta_sigma for MoCo')
    group.add_argument('--MoCo_gamma', type=float, default=0.1, help='gamma for MoCo')
    group.add_argument('--MoCo_gamma_sigma', type=float, default=0.5, help='MoCo_gamma_sigma for MoCo')
    group.add_argument('--MoCo_rho', type=float, default=0, help='MoCo_rho for MoCo')
    return parser


def build_weighting(opts, model):
    kwargs = {}
    if opts['weighting'] == 'none':
        weighting = weighting_method.__dict__['EW']()
        weighting.share_net = model.attention
        weighting.rep_grad = False
        weighting.device = opts['gpu_index'][0]
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
        weighting.share_net = model.attention
        weighting.rep_grad = False
        weighting.device = opts['gpu_index'][0]
        weighting.task_num = opts['task_num']
        weighting.init_param()
        return weighting, kwargs
    else:
        raise ValueError('No support weighting method {}'.format(opts['weighting'])) 