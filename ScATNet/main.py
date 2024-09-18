from config.build import *
from experiment.experiment import *
import json
from config.opts import get_config
from utilities.util import save_arguments
from os.path import basename, dirname, splitext
import pdb

def main(args):
    # configuration for experiment


    # -----------------------------------------------------------------------------
    # Preparing Dataset
    # -----------------------------------------------------------------------------
    seed_everything(args)
    train_loader, valid_loader, train_valid_loader, test_loader = build_dataset(args)
    args = build_class_weights(args)


    # -----------------------------------------------------------------------------
    # Model
    #   - setup model
    #   - load state dict if resume is chosen
    #   - gpu setup and data parallel
    # -----------------------------------------------------------------------------
    args = build_cuda(args)
    criterion = build_criteria(args)
    model, feature_extractor = build_model(args)
    if args['attn_guide']:
        attn_criterion = build_attn_criteria(args)
    else:
        attn_criterion = None

    if args['tissue_constraint']:
        args_copy = args.copy()
        args_copy['attn_layer'] = [0,1,2,3]
        args_copy['select_attn_head'] = 'fixed'
        args_copy['attn_head'] = 4
        args_copy['attn_loss'] = 'in-out'
        tissue_criterion = build_attn_criteria(args_copy)
    else:
        tissue_criterion = None


    # -----------------------------------------------------------------------------
    # Experiment Setup
    #   - setup visdom and logger
    #   - calculate class weights, setup loss function
    #   - setup optimizer, scheduler
    # -----------------------------------------------------------------------------

    args = build_visualization(args)
    engine = experiment_engine(train_loader, valid_loader, train_valid_loader,
                               test_loader, **args)
    mtl_weighting, mtl_kwargs = build_weighting(args, model)
    if args['mode'] != 'kmeans':
        optimizer = build_optimizer(args, model, mtl_weighting)
        scheduler = build_scheduler(args, optimizer)

    # -----------------------------------------------------------------------------
    # Training and Evaluation
    # -----------------------------------------------------------------------------

    if args['mode'] == 'train' or args['mode'] == 'train-on-train-valid':
        print_info_message('Training Process Starts...')
        print_info_message("Number of Parameters: {:.2f} M".format(sum([p.numel() for p in model.parameters()])/1e6))
        result = engine.train(model, args['epochs'], criterion, mtl_weighting, mtl_kwargs,
                     optimizer, scheduler,
                     args['start_epoch'], feature_extractor=feature_extractor,
                              criterion_attn=attn_criterion, criterion_tissue=tissue_criterion)
    elif args['mode'] == 'test':
        print_info_message('Evaluation on Test Process Starts...')
        result = engine.eval(model, criterion, mode='test',
                    feature_extractor=feature_extractor)
    elif args['mode'] == 'valid':
        print_info_message('Evaluation on Validation Process Starts...')
        result = engine.eval(model, criterion, mode='val', feature_extractor=feature_extractor)
    elif args['mode'] == 'test-on-train':
        print_info_message('Evaluation on Training Process Starts...')
        result = engine.eval(model, criterion, mode= 'train', feature_extractor=feature_extractor)
    elif args['mode'] == 'test-on-train-valid':
        print_info_message('Evaluation on Training and Valid Process Starts...')
        result = engine.eval(model, criterion, mode= 'train-on-train-valid', feature_extractor=feature_extractor)
    return result


if __name__ == '__main__':
    opts, parser = get_config()
    if opts.resize1 is None:
        resize1 = ['real', 'real']
    else:
        resize1 = opts.resize1
    argument_fname = 'config_{}_{}_{}'.format(basename(dirname(opts.data)),
                                                    opts.model, opts.mode)

    opts.save_name = '{}scale_{}_{}x{}_dropout{}'.format(len(opts.resize1_scale),
                                                            basename(dirname(opts.data)),
                                                            opts.model_dim,
                                                            opts.n_layers,
                                                            opts.drop_out)

    save_arguments(args=opts, save_loc=opts.model_dir, json_file_name=argument_fname)
    print_log_message('Arguments')
    print(json.dumps(vars(opts), indent=4, sort_keys=True))
    main(vars(opts))

