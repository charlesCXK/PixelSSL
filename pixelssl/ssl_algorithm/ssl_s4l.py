import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from pixelssl.utils import REGRESSION, CLASSIFICATION
from pixelssl.utils import logger, tool
from pixelssl.nn import func
from pixelssl.nn.module import patch_replication_callback

from . import ssl_base


""" Implementation of pixel-wise self-supervised semi-supervised learning (S4L)
    
This method is proposed in paper: 
    'S4L: Self-Supervised Semi-Supervised Learning'
This implementation only supports the rotation-based self-supervised pretext task.
"""


def add_parser_arguments(parser):
    ssl_base.add_parser_arguments(parser)
    parser.add_argument('--rotation-scale', type=float, default=-1, help='rotation-based self-supervised coefficient')


def ssl_s4l(args, model_dict, optimizer_dict, lrer_dict, criterion_dict, task_func):
    if not len(model_dict) == len(optimizer_dict) == len(lrer_dict) == len(criterion_dict) == 1:
        logger.log_err('The len(element_dict) of SSL_S4L should be 1\n')
    elif list(model_dict.keys())[0] != 'model':
        logger.log_err('In SSL_S4L, the key of element_dict should be \'model\',\n'
                'but \'{0}\' is given\n'.format(model_dict.keys()))

    model_funcs = [model_dict['model']]
    optimizer_funcs = [optimizer_dict['model']]
    lrer_funcs = [lrer_dict['model']]
    criterion_funcs = [criterion_dict['model']]

    algorithm = SSLS4L(args)
    algorithm.build(model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func)
    return algorithm


class SSLS4L(ssl_base._SSLBase):
    NAME = 'ssl_s4l'
    SUPPORTED_TASK_TYPES = [REGRESSION, CLASSIFICATION]

    def __init__(self, args):
        super(SSLS4L, self).__init__(args)

        # define the task model and the FC discriminator
        self.model = None
        self.optimizer = None
        self.lrer = None
        self.criterion = None

        # check SSL arguments
        if self.args.rotation_scale < 0:
            logger.log_err('The argument - rotation_scale - is not set (or invalid)\n'
                           'Please set - rotation_scale >= 0 - for training\n')


    def _build(self, model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func):
        self.task_func = task_func

        # create models
        self.model = func.create_model(model_funcs[0], 'model', args=self.args)
        # call 'patch_replication_callback' to enable the `sync_batchnorm` layer
        patch_replication_callback(self.model)
        self.models = {'model': self.model}

        # create optimizers
        self.optimizer = optimizer_funcs[0](self.model.module.param_groups)
        self.optimizers = {'optimizer': self.optimizer}

        # create lrers
        self.lrer = lrer_funcs[0](self.optimizer)
        self.lrers = {'lrer': self.lrer}

        # create criterions
        self.criterion = criterion_funcs[0](self.args)
        self.criterions = {'criterion': self.criterion}

        self._algorithm_warn()

    def _train(self, data_loader, epoch):
        self.meters.reset()
        lbs = self.args.labeled_batch_size

        self.model.train()

        # both 'inp' and 'gt' are tuples
        for idx, (inp, gt) in enumerate(data_loader):
            timer = time.time()

            inp, gt = self._batch_prehandle(inp, gt)
            if len(gt) > 1 and idx == 0:
                self._inp_warn()

            self.optimizer.zero_grad()

            # forward the task model
            resulter, debugger = self.model.forward(inp)
            if not 'pred' in resulter.keys() or not 'activated_pred' in resulter.keys():
                self._pred_err()

            pred = tool.dict_value(resulter, 'pred')
            activated_pred = tool.dict_value(resulter, 'activated_pred')

            # calculate the supervised task constraint on the labeled data
            l_pred = func.split_tensor_tuple(pred, 0, lbs)
            l_gt = func.split_tensor_tuple(gt, 0, lbs)
            l_inp = func.split_tensor_tuple(inp, 0, lbs)

            # 'task_loss' is a tensor of 1-dim & n elements, where n == batch_size
            task_loss = self.criterion.forward(l_pred, l_gt, l_inp)
            task_loss = torch.mean(task_loss)
            self.meters.update('task_loss', task_loss.data)

            loss = task_loss
            loss.backward()
            self.optimizer.step()

            # logging
            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}\n'
                                '  task-{3}\t=>\t'
                                'task-loss: {meters[task_loss]:.6f}\n'
                                # '  rotation-{3}\t=>\t'
                                # 'rotation-loss: {meters[rotation_loss]:.6f}\t'
                                # 'rotation-acc: {meters[rotation_acc]:.6f}\n'
                                .format(epoch, idx, len(data_loader), self.args.task, meters=self.meters))
                
            # visualization
            if self.args.visualize and idx % self.args.visual_freq == 0:
                self._visualize(epoch, idx, True, 
                                func.split_tensor_tuple(inp, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(activated_pred, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(gt, 0, 1, reduce_dim=True))

            # update iteration-based lrers
            if not self.args.is_epoch_lrer:
                self.lrer.step()
        
        # update epoch-based lrers
        if self.args.is_epoch_lrer:
            self.lrer.step()

    def _validate(self, data_loader, epoch):
        self.meters.reset()
        
        self.model.eval()

        for idx, (inp, gt) in enumerate(data_loader):
            timer = time.time()

            inp, gt = self._batch_prehandle(inp, gt)
            if len(gt) > 1 and idx == 0:
                self._inp_warn()
            
            resulter, debugger = self.model.forward(inp)
            if not 'pred' in resulter.keys() or not 'activated_pred' in resulter.keys():
                self._pred_err()
            
            pred = tool.dict_value(resulter, 'pred')
            activated_pred = tool.dict_value(resulter, 'activated_pred')

            task_loss = self.criterion.forward(pred, gt, inp)
            task_loss = torch.mean(task_loss)
            self.meters.update('task_loss', task_loss.data)

            self.task_func.metrics(activated_pred, gt, inp, self.meters, id_str='task')
            
            # logging
            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}\n'
                                '  task-{3}\t=>\t'
                                'task-loss: {meters[task_loss]:.6f}\n'
                                # '  rotation-{3}\t=>\t'
                                # 'rotation-loss: {meters[rotation_loss]:.6f}\t'
                                # 'rotation-acc: {meters[rotation_acc]:.6f}\n'
                                .format(epoch, idx, len(data_loader), self.args.task, meters=self.meters))

            # visualization
            if self.args.visualize and idx % self.args.visual_freq == 0:
                self._visualize(epoch, idx, False, 
                                func.split_tensor_tuple(inp, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(activated_pred, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(gt, 0, 1, reduce_dim=True))
        # metrics
        metrics_info = {'task': ''}
        for key in sorted(list(self.meters.keys())):
            if self.task_func.METRIC_STR in key:
                for id_str in metrics_info.keys():
                    if key.startswith(id_str):
                        metrics_info[id_str] += '{0}: {1:.6}\t'.format(key, self.meters[key])
            
        logger.log_info('Validation metrics:\n task-metrics\t=>\t{0}\n'.format(metrics_info['task'].replace('_', '-')))

    def _save_checkpoint(self, epoch):
        state = {
            'algorithm': self.NAME,
            'epoch': epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lrer': self.lrer.state_dict(),
        }

        checkpoint = os.path.join(self.args.checkpoint_path, 'checkpoint_{0}.ckpt'.format(epoch))
        torch.save(state, checkpoint)

    def _load_checkpoint(self):
        checkpoint = torch.load(self.args.resume)

        checkpoint_algorithm = tool.dict_value(checkpoint, 'algorithm', default='unknown')
        if checkpoint_algorithm != self.NAME:
            logger.log_err('Unmatched ssl algorithm format in checkpoint => required: {0} - given: {1}\n'
                           .format(self.NAME, checkpoint_algorithm))

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lrer.load_state_dict(checkpoint['lrer'])

        return checkpoint['epoch']
        
    # -------------------------------------------------------------------------------------------
    # Tool Functions for the SSL_S4L Framework
    # -------------------------------------------------------------------------------------------

    def _visualize(self, epoch, idx, is_train, inp, pred, gt):
        visualize_path = self.args.visual_train_path if is_train else self.args.visual_val_path
        out_path = os.path.join(visualize_path, '{0}_{1}'.format(epoch, idx))

        self.task_func.visualize(out_path, id_str='task', inp=inp, pred=pred, gt=gt)
        
    def _batch_prehandle(self, inp, gt):
        # add extra data augmentation process here if necessary
        
        inp_var = []
        for i in inp:
            inp_var.append(Variable(i).cuda())
        inp = tuple(inp_var)
            
        gt_var = []
        for g in gt:
            gt_var.append(Variable(g).cuda())
        gt = tuple(gt_var)

        return inp, gt

    def _algorithm_warn(self):
        logger.log_warn('This SSL_S4L algorithm reproducts the SSL algorithm from paper:\n'
                        '  \'S4L: Self-Supervised Semi-Supervised Learning\'\n'
                        'The main differences between this implementation and the original paper are:\n'
                        '  (1) This is an implementation for pixel-wise vision tasks\n'
                        '  (2) This implementation only supports the 4-angle (0, 90, 180, 270) rotation-based self-supervised pretext task\n')


    def _inp_warn(self):
        logger.log_warn('More than one ground truth of the task model is given in SSL_S4L\n'
                        'You try to train the task model with more than one (pred & gt) pairs\n'
                        'Please make sure that:\n'
                        '  (1) The prediction tuple has the same size as the ground truth tuple\n'
                        '  (2) The elements with the same index in the two tuples are corresponding\n'
                        '  (3) All elements in the ground truth tuple should be 4-dim tensors since S4L\n'
                        '      will rotate them to match the rotated inputs\n'
                        'Please implement a new SSL algorithm if you want a variant of SSL_S4L that\n' 
                        'supports other formants (not 4-dim tensor) of the ground truth\n')

    def _pred_err(self):
        logger.log_err('In SSL_S4L, the \'resulter\' dict returned by the task model should contain the following keys:\n'
                       '   (1) \'pred\'\t=>\tunactivated task predictions\n'
                       '   (2) \'activated_pred\'\t=>\tactivated task predictions\n'
                       'We need both of them since some losses include the activation functions,\n'
                       'e.g., the CrossEntropyLoss has contained SoftMax\n')
