import tqdm
import torch
from utilities.print_fc import print_report
from sklearn.metrics import accuracy_score
import os
from torch import nn
from experiment.EMA import EMA
from utilities.visualize import compute_stats, compute_case, visualize_top_k_crop, plot_weights
from dataset.dataloaders import sliding_dataloader
import numpy as np
import json
import pdb

class experiment_engine(object):
    def __init__(self, train_loader,
                 val_loader, train_val_loader, test_loader, **args):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_val_loader = train_val_loader
        self.test_loader = test_loader
        self.__dict__.update(**args)
        # moving average of all
        self.EMA1 = None
        self.kd_weight = 0

    def multi_scale_features(self, multi_data, mask, labels_conf, feature_extractor):
        multi_feat = []
        num_scales = len(multi_data)
        for j in range(num_scales):
            _data = multi_data[j]
            if feature_extractor is not None:
                split_size = self.max_bsz_cnn_gpu0 if self.max_bsz_cnn_gpu0 > 0 else len(_data)
                split_data = torch.split(_data, split_size_or_sections=split_size, dim=1)

                feat = []
                for data in split_data:
                    bsz, n_split, channels, width, height = data.size()
                    if self.use_gpu:
                        data = data.to(device=self.gpu_id[0])
                    # reshape
                    data = data.contiguous().view(bsz * n_split, channels, width, height)

                    if self.finetune_base_extractor and self.training:
                        data = feature_extractor(data)
                        data = data.contiguous().view(bsz, n_split, -1)
                    else:
                        feature_extractor.eval()
                        with torch.no_grad():
                            data = feature_extractor(data)
                            data = data.contiguous().view(bsz, n_split, -1)

                    feat.append(data)

                feat = torch.cat(feat, dim=1)
                multi_feat.append(feat)

            if self.use_gpu:
                if self.mask_type == 'return-indices' and self.mask is not None:
                    mask = [m.to(self.gpu_id[0]) for m in mask]
                else:
                    mask = None
                labels_conf = labels_conf.to(self.gpu_id[0])
        return multi_feat, mask, labels_conf

    def train(self, model, epochs, criterion, mtl_weighting, mtl_args,
              optimizer, scheduler=None,
              start_epoch=0, feature_extractor=None, **kwargs):

        model.train()
        val_loss_min = 10000
        val_acc_max = -1
        step = 0
        eval_stats_dict = dict()
        self.EMA1 = EMA(model, ema_momentum=0.001)
        use_attn_guide = self.attn_guide
        tissue_attn_guide = self.tissue_constraint
        criterion_attn = kwargs.get('criterion_attn', None)
        lambda_attn = self.lambda_attn
        criterion_tissue = kwargs.get('criterion_tissue', None)
        lambda_tissue = self.lambda_tissue

        dataloader = self.train_loader

        self.batch_weight = np.zeros([self.task_num, epochs, len(dataloader)+1])

        for epoch in range(start_epoch, epochs):
            model.train()
            if self.visdom:
                self.logger.update(epoch, optimizer.param_groups[0]['lr'], mode='lr')
            if self.attn_schedule:
                lambda_attn = self.lambda_attn * (1 - (epoch - start_epoch)/(epochs - start_epoch))
            elif self.attn_schedule_reversed:
                lambda_attn = self.lambda_attn * (epoch - start_epoch)/(epochs - start_epoch)
            epoch_loss = 0
            output = []
            scores = []
            target = []
            optimizer.zero_grad()
            for i, (multi_data, labels, labels_conf, paths, attn_mask, tissue_attn_mask) in tqdm.tqdm(enumerate(dataloader), leave=False, total=len(dataloader)):
                step += 1
                if self.warmup and step < 500:
                    lr_scale = min(1., float(step + 1) / 500.)
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr_scale * self.lr
                elif step > 500 and self.scheduler == 'cosine':
                    scheduler.step(epoch)
                elif self.scheduler == 'cycle':
                    scheduler.step()
                else:
                    scheduler.step(epoch)
                target.extend(l.item() for l in labels)
                if multi_data[0].dim() != 3:
                    # [B x C x 3 x H x W] x Scales
                    multi_feat, mask, labels_conf = self.multi_scale_features(multi_data=multi_data,
                                                                          feature_extractor=feature_extractor,
                                                                          mask=mask, labels_conf=labels_conf)
                    attn_mask = [s.float().to(self.gpu_id[0]) for s in attn_mask]
                    tissue_attn_mask = [s.float().to(self.gpu_id[0]) for s in tissue_attn_mask]
                else:
                    # [B x C x F] x Scales
                    labels_conf = labels_conf.to(device=self.gpu_id[0])
                    multi_feat = [d.to(device=self.gpu_id[0]) for d in multi_data]
                    mask=None
                    attn_mask = [m.float().to(device=self.gpu_id[0]) for m in attn_mask]
                    tissue_attn_mask = [m.float().to(device=self.gpu_id[0]) for m in tissue_attn_mask]

                out, _, _, attn_over_layers = model(x=multi_feat, src_mask=mask)

                # use teacher model to adjust the soft label
                if self.label_update != None and self.kd_weight > 0:
                    teacher_out, _, _, _ = self.EMA1.ema_model(x=multi_feat, src_mask=mask)
                    teacher_out = nn.Softmax(dim=-1)(teacher_out).detach()
                    labels_conf = (1 - self.kd_weight) * labels_conf + self.kd_weight * teacher_out
                
                # Aggregate Loss
                losses = torch.zeros(self.task_num).to(self.gpu_id[0])
                losses[0] = criterion(out, labels_conf)
                cur_task = 1
                if use_attn_guide:
                    attn_guide_loss = criterion_attn(attn_over_layers, attn_mask, bs=len(labels))
                    if torch.isnan(attn_guide_loss):
                        pdb.set_trace()
                    losses[cur_task] = lambda_attn * attn_guide_loss
                    cur_task += 1
                if tissue_attn_guide:
                    tissue_attn_loss = criterion_tissue(attn_over_layers, tissue_attn_mask, bs=len(labels))
                    if torch.isnan(tissue_attn_loss):
                        pdb.set_trace()
                    losses[cur_task] = lambda_tissue * tissue_attn_loss
                    cur_task += 1
                
                # loss backward (MTL)
                w = mtl_weighting.backward(losses, device=self.gpu_id[0], **mtl_args)
                if w is not None:
                    self.batch_weight[:, epoch, i] = w
                
                if (i + 1) % self.aggregate_batch == 0 or (i + 1) == len(dataloader):
                    optimizer.step()
                    self.EMA1.update_parameters(model)
                    optimizer.zero_grad()

                scores.append(out.detach().cpu())
                epoch_loss += torch.mul(losses.cpu().detach(), torch.Tensor(w)).sum()
                _, predicted = torch.max(out.data, 1)
                output.extend(predicted.detach().cpu().tolist())
                if self.visdom:
                    self.confusion_meter.add(predicted, labels)
            if self.visdom:
                self.logger.update(epoch, epoch_loss/len(dataloader), mode='train_loss')
                train_acc = accuracy_score(target, output)
                self.logger.update(epoch, train_acc, mode='train_err')
                self.logger.update(epoch, self.confusion_meter.value(), mode='train_mat')

            else:
                print('loss: {:0.23f}'.format(epoch_loss/len(dataloader)))
                print_report(output, target, name='Train', epoch=epoch)

            val_loss, val_acc, summary, slice_acc = self.eval(model, criterion,
                                 epoch=epoch, mode='train-on-train-valid' if self.mode!='train' else 'val', feature_extractor=feature_extractor)
            if step > 500 and self.scheduler == 'reduce':
                scheduler.step(val_loss)
            ema_val_loss, ema_val_acc, ema_summary, ema_slice_acc = self.eval(self.EMA1.ema_model, criterion, 
                                                               epoch=epoch, mode='train-on-train-valid'  if self.mode!='train' else 'val', feature_extractor=feature_extractor)
            eval_stats_dict[epoch] = {'cur_iter': summary, 'ema': ema_summary}
            if self.label_update != None:
                self.update_kd_weight(epoch, slice_acc, ema_slice_acc)

            if val_acc >= val_acc_max:
                print('Valid acc increased ({:.6f} --> {:.6f}).  Saving model...'.format(val_acc_max, val_acc))
                self.save_model(model, 'best', val_acc)
            self.save_model(model, epoch, val_acc)
            if val_acc > val_acc_max:
                val_acc_max = val_acc
            elif val_loss < val_loss_min:
                val_loss_min = val_loss

        if self.visdom:
            self.logger.close()

        eval_stats_fname = '{}/val_stats_{}'.format(self.savedir, self.save_name)
        if not os.path.isfile(eval_stats_fname):
            with open(eval_stats_fname, 'w') as json_file:
                json.dump(eval_stats_dict, json_file)
        else:
            with open(eval_stats_fname, 'r') as json_file:
                eval_stats_dict_old = json.load(json_file)
            eval_stats_dict_old.update(eval_stats_dict)
            with open(eval_stats_fname, 'w') as json_file:
                json.dump(eval_stats_dict, json_file)
        
        weight_fname = '{}/{}_weights.npy'.format(self.savedir, self.weighting)
        np.save(weight_fname, self.batch_weight)
        plot_weights(self.batch_weight, self.savedir)

    def update_kd_weight(self, epoch, cur_acc=0, ema_acc=0):
        '''knowledge distillation weight update'''
        if epoch < 20:
            self.kd_weight = 0
        elif self.label_update == 'scheduled':
            self.kd_weight = 1 - np.exp(-0.005 * epoch)
        elif self.label_update == 'conf':
            self.kd_weight = ema_acc / (cur_acc + ema_acc)
        else:
            self.kd_weight = 0
        print('KD weight adjust to {:.6f}'.format(self.kd_weight))

    def save_model(self, model, epoch, loss):
        from pathlib import Path
        if epoch == 'EMA':
            d = os.path.join(self.model_dir, 'EMA')
        else:
            d = self.model_dir
        save_path = os.path.join(d, '{}_{}_{}.pt'.format(self.save_name, epoch, loss))


        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)


    def calc_gradient(self, model, data, mask, save_name):
        # batch_size in data must be one
        # save_name should be a string
        if len(data.shape) < 4:
            data.unsqueeze(0)
        if len(mask.shape) == 1:
            mask=None
        elif len(mask.shape) < 4:
            mask.unsqueeze(0)
        data.requires_grad = True
        with torch.enable_grad():
            output,_ = model({'image': data, 'mask': mask, 'save_name':save_name})
            grad = torch.autograd.grad(output.max(1)[0],
                                       data,
                                       only_inputs=True,
                                       allow_unused=False, create_graph=False)
            # BCHW --> CHW
            grad = grad[0].squeeze(0)
            # CHW --> HW
            #min_val = torch.min(grad)
            #grad += abs(min_val)
            grad = grad ** 2
            grad = torch.mean(grad, dim=0)
            grad = torch.sqrt(grad)
            # min-max normalization
            min_val = torch.min(grad)
            max_val = torch.max(grad)
            grad = torch.add(grad, -min_val)
            grad = torch.div(grad, max_val - min_val)
            grad *= 255.0
            grad = grad.byte().cpu().numpy()


        _, predicted = torch.max(output.data, 1)
        result = predicted.detach().cpu().item()
        return grad, output, result


    def eval(self, model, criterion,
             epoch=None, mode='val', feature_extractor=None):
        return self.eval_crop(model, criterion, epoch=epoch,
                                mode=mode, feature_extractor=feature_extractor)

    
    def eval_crop(self, model, criterion,
                  epoch=None, mode='val', feature_extractor=None):
        if mode == 'train-on-train-valid':
            dataloader = self.train_val_loader
        elif 'val' in mode or 'ema' in mode:
            dataloader = self.val_loader
        elif 'test' in mode:
            dataloader = self.test_loader
        elif 'train' in mode:
            dataloader = self.train_loader
        else:
            import sys
            sys.exit('Wrong evaluation mode. Choices are val or test')

        model.eval()
        val_target = []
        val_output = []
        val_loss = 0
        scores = []
        results_list = []
        if self.visdom:
            self.confusion_meter.reset()
        with torch.no_grad():
            for i, (multi_data, target, target_conf, paths, attn_mask, tissue_attn_map) in tqdm.tqdm(enumerate(dataloader),
                                                                               leave=False,
                                                                               total=max(1, len(dataloader))):
                val_target.extend(t.item() for t in target)
                labels_conf = target_conf.to(device=self.gpu_id[0])

                if multi_data[0].dim() != 3:
                    # [B x C x 3 x H x W] x Scales
                    multi_feat, mask, labels_conf = self.multi_scale_features(multi_data=multi_data,
                                                                          feature_extractor=feature_extractor,
                                                                          mask=mask, labels_conf=labels_conf)
                else:
                    # [B x C x F] x Scales
                    multi_feat = [d.to(device=self.gpu_id[0]) for d in multi_data]
                    mask = None

                output, output_vis, scale_attn, _ = model(x=multi_feat, src_mask=mask)

                if self.loss_function != 'bce':
                    probabilities = nn.Softmax(dim=-1)(output.detach().cpu())
                else:
                    probabilities = nn.Sigmoid()(output.detach().cpu())
                scores.append(output.detach().cpu())
                val_loss += criterion(output, labels_conf).detach().cpu().item()
                _, predicted = torch.max(output.data, 1)
                val_output.extend(predicted.detach().cpu().tolist())
                for j in range(len(paths)):
                    results_list.append((paths[j], int(target[j].item()), int(predicted[j].item())))
                if self.save_top_k > 0:
                    visualize_top_k_crop(multi_data, self.save_top_k,
                                         predicted, target, output_vis,
                                         paths, self.savedir,
                                         multi_scale=self.multi_scale >= 1,
                                         mask=mask)
                if self.save_result:
                    for j in range(len(paths)):
                        with open(os.path.join(self.savedir, '{}_result.txt'.format(mode)), 'a') as f:
                            f.write('{}; {}; {}; {}\n'.format(paths[j], int(predicted[j].item()),
                                                          probabilities[j].tolist(), ''))

                if self.visdom:
                    self.confusion_meter.add(predicted, target)
        scores = torch.cat(scores, dim=0)
        scores = scores.float()
        if self.loss_function == 'bce':
            predictions_max_sm = nn.Sigmoid()(scores)
        else:
            predictions_max_sm = nn.Softmax(dim=-1)(scores)  # Image x Classes
        pred_label_max1 = torch.max(predictions_max_sm, dim=-1)[1]  # Image x 1
        pred_label_max = pred_label_max1.byte().cpu().numpy().tolist()  # Image x 1
        pred_conf_max = predictions_max_sm.float().cpu().numpy()  # Image x Classes
        # val_acc = accuracy_score(val_target, val_output)
        
        # val_acc, results_summary = compute_case(results_list, pred_conf_max, verbose=(epoch is None), mode=mode,
        #                            savepath=self.savedir, save=self.save_result)
        if self.mode == 'test' or self.mode == 'valid':
            sp = self.savedir
        else:
            sp = None
        
        val_acc, results_summary = compute_case(results_list, pred_conf_max, verbose=(epoch is None), mode=mode,
                                   savepath=self.savedir, save=self.save_result)
        slice_acc = accuracy_score(val_target, val_output)
        if self.visdom:
            self.logger.update(epoch, val_loss / max(1,len(dataloader)), mode='{}_loss'.format(mode))
            self.logger.update(epoch, val_acc, mode='{}_err'.format(mode))
            self.logger.update(epoch, self.confusion_meter.value(), mode='{}_mat'.format(mode))
        else:
            compute_stats(y_true=val_target, y_pred=pred_label_max, y_prob=pred_conf_max, logger=None,
                          mode='{}_roc'.format(mode), num_classes=self.num_classes,
                          savepath=sp, fname=self.save_name)
            name = 'Valid' if 'val' in mode else 'Test'
            print_report(val_output, val_target, name=name, epoch=epoch)
        return val_loss / max(1, len(dataloader)), val_acc, results_summary, slice_acc


