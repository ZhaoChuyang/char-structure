import sys
import json
import time
import argparse
import pickle
from pprint import pprint
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from src.preprocess.peanut import PeanutDataset, get_heatmap, get_paf, Stroke
from scipy.io import savemat, loadmat
import src.preprocess.util as util
import torch
from torch.optim import lr_scheduler
from src.preprocess.peanut import PeanutDataset, PeanutClsDataset, OpenPoseDataset, OriOpenPoseDataset
from torch.utils.data import DataLoader
from src.modeling.char_structure import bodypose_model, char_classifier
from src.modeling.refinement_net import RefinementNet, BackboneNet
from src.modeling.openpose import OpenPose, SimpleOpenPose, SimpleNet, SimpleClassifier
from src.modeling.utils.loss import Loss
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


try:
    from apex import amp
except ImportError:
    print('not import apex')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test', 'view'])
    parser.add_argument('model', choices=['cls', 'refinement', 'base', 'openpose'])
    parser.add_argument('--dataset', type=str, default='data/images')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--step-size', type=int, default=30)
    parser.add_argument('--train-all', action='store_true')
    parser.add_argument('--snapshot', type=str, default=None)
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--droprate', type=float, default=0.5, help='drop rate')
    parser.add_argument('--warm-epoch', default=0, type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--apex', action='store_true')
    parser.add_argument('--num-classes', type=int)
    return parser.parse_args()


def save_test_result(args, model, type, test_dataloader):
    model.eval()
    if args.snapshot:
        util.load_model(args.snapshot, model)
    for id, (inputs, heatmaps, paf_maps, char_ids, ids, image_names) in enumerate(test_dataloader):
        if args.use_gpu:
            inputs = inputs.cuda()
        gt_heatmaps = heatmaps.cpu().numpy()
        gt_paf_maps = paf_maps.cpu().numpy()
        if type == 'base':
            output_paf_maps, output_heatmaps = model(inputs)
            output_heatmaps = output_heatmaps.cpu().numpy()
            output_paf_maps = output_paf_maps.cpu().numpy()
        if type == 'refinement':
            outputs = model(inputs)
            output_heatmaps = (outputs[0].cpu().numpy() + outputs[2].cpu().numpy()) / 2
            output_paf_maps = (outputs[1].cpu().numpy() + outputs[3].cpu().numpy()) / 2

        for img_id, image_name, paf_map, heatmap, gt_heatmap, gt_paf_map in zip(ids, image_names, output_paf_maps, output_heatmaps, gt_heatmaps, gt_paf_maps):
            result = {
                'image_name': image_name,
                'paf_map': paf_map,
                'heatmap': heatmap,
                'gt_heatmap': gt_heatmap,
                'gt_paf_map': gt_paf_map,
            }
            savemat('output/%s_result.mat' % image_name, result)
        break


def save_openpose_result(args, model, type, test_dataloader):
    model.eval()
    if args.snapshot:
        util.load_model(args.snapshot, model)
    for id, (data_dict) in enumerate(test_dataloader):
        if args.use_gpu:
            data_dict['img'] = data_dict['img'].cuda()
            data_dict['heatmap'] = data_dict['heatmap'].cuda()
            data_dict['vecmap'] = data_dict['vecmap'].cuda()
            data_dict['maskmap'] = data_dict['maskmap'].cuda()
        else:
            raise
        gt_heatmaps = data_dict['heatmap'].cpu().numpy()
        gt_paf_maps = data_dict['vecmap'].cpu().numpy()
        filenames = data_dict['filename']

        outputs = model(data_dict)
        output_paf_maps = outputs['paf'].cpu().numpy()
        output_heatmaps = outputs['heatmap'].cpu().numpy()
        inputs = data_dict['img'].cpu().numpy()

        for image_name, paf_map, heatmap, gt_heatmap, gt_paf_map, input in zip(filenames, output_paf_maps, output_heatmaps, gt_heatmaps, gt_paf_maps, inputs):
            result = {
                'image_data': input,
                'image_name': image_name,
                'paf_map': paf_map,
                'heatmap': heatmap,
                'gt_heatmap': gt_heatmap,
                'gt_paf_map': gt_paf_map,
            }
            savemat('output/%s_result.mat' % image_name, result)

        break


def train(args, model, train_dataloader, valid_dataloader):
    params = model.parameters()
    optim = torch.optim.SGD([{'params': params, 'lr': args.lr}], weight_decay=5e-4, momentum=0.9, nesterov=True)
    if args.apex:
        amp.initialize(model, optim, opt_level='O1')
    # FIXME: Add mask if useful
    criterion = torch.nn.MSELoss()

    best = {
        'loss': float('inf'),
        'epoch': -1,
        'warm_up': 0.1,
    }

    if args.snapshot:
        if args.apex:
            detail = util.load_model(args.resume_from, model, optim=optim, amp=amp)
        else:
            detail = util.load_model(args.resume_from, model, optim=optim)
        best.update({
            'loss': detail['loss'],
            'epoch': detail['epoch'],
            'warm_up': detail['warm_up'],
        })

    warm_up = best['warm_up']
    warm_iteration = len(train_dataloader) * args.warm_epoch
    scheduler = lr_scheduler.StepLR(optim, step_size=args.step_size, gamma=0.1)
    for epoch in range(best['epoch'] + 1, args.epoch):
        print(f'\n----- epoch {epoch} -----')
        util.set_seed(epoch)
        if epoch < args.warm_epoch:
            warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
            run_nn(args, 'train', model, train_dataloader, warm_up=warm_up, criterion=criterion, optim=optim,
                   apex=args.apex)
        else:
            run_nn(args, 'train', model, train_dataloader, criterion=criterion, optim=optim,
                   apex=args.apex)
        with torch.no_grad():
            val = run_nn(args, 'valid', model, valid_dataloader, criterion=criterion)

        detail = {
            'loss': val['loss'],
            'epoch': epoch,
            'warm_up': warm_up,
        }
        if val['loss'] <= best['loss']:
            best.update(detail)
        if epoch % 5 ==0:
            if args.apex:
                util.save_model(model, optim, 'base_model_apex', detail, amp=amp)
            else:
                util.save_model(model, optim, 'base_model', detail)
        print('[best] ep:%d loss:%.4f' % (best['epoch'], best['loss']))

        scheduler.step()


def train_refinement_net(args, model, train_dataloader, valid_dataloader):
    params = model.parameters()
    # optim = torch.optim.SGD([{'params': params, 'lr': args.lr}], weight_decay=5e-4, momentum=0.9, nesterov=True)
    optim = torch.optim.SGD([{'params': params, 'lr': args.lr}])
    if args.apex:
        amp.initialize(model, optim, opt_level='O1')
    # FIXME: Add mask if useful
    criterion = torch.nn.MSELoss()

    best = {
        'loss': float('inf'),
        'epoch': -1,
        'warm_up': 0.1,
    }

    if args.snapshot:
        if args.apex:
            detail = util.load_model(args.resume_from, model, optim=optim, amp=amp)
        else:
            detail = util.load_model(args.resume_from, model, optim=optim)
        best.update({
            'loss': detail['loss'],
            'epoch': detail['epoch'],
            'warm_up': detail['warm_up'],
        })

    warm_up = best['warm_up']
    warm_iteration = len(train_dataloader) * args.warm_epoch
    scheduler = lr_scheduler.StepLR(optim, step_size=args.step_size, gamma=0.1)
    for epoch in range(best['epoch'] + 1, args.epoch):
        print(f'\n----- epoch {epoch} -----')
        util.set_seed(epoch)
        if epoch < args.warm_epoch:
            warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
            run_refinement_nn(args, 'train', model, train_dataloader, warm_up=warm_up, criterion=criterion, optim=optim,
                   apex=args.apex)
        else:
            run_refinement_nn(args, 'train', model, train_dataloader, criterion=criterion, optim=optim,
                   apex=args.apex)
        with torch.no_grad():
            val = run_refinement_nn(args, 'valid', model, valid_dataloader, criterion=criterion)

        detail = {
            'loss': val['loss'],
            'epoch': epoch,
            'warm_up': warm_up,
        }
        if val['loss'] <= best['loss']:
            best.update(detail)
        if (epoch+1) % 5 ==0:
            if args.apex:
                util.save_model(model, optim, 'refinement_model_apex', detail, amp=amp)
            else:
                util.save_model(model, optim, 'refinement_model', detail)
        print('[best] ep:%d loss:%.4f' % (best['epoch'], best['loss']))

        scheduler.step()


def train_cls_net(args, model, train_dataloader, valid_dataloader):
    params = model.parameters()
    optim = torch.optim.SGD([{'params': params, 'lr': args.lr}], weight_decay=5e-4, momentum=0.9, nesterov=True)
    if args.apex:
        amp.initialize(model, optim, opt_level='O1')
    # FIXME: Add mask if useful
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCEWithLogitsLoss()

    best = {
        'loss': float('inf'),
        'epoch': -1,
        'warm_up': 0.1,
    }

    if args.snapshot:
        if args.apex:
            detail = util.load_model(args.resume_from, model, optim=optim, amp=amp)
        else:
            detail = util.load_model(args.resume_from, model, optim=optim)
        best.update({
            'loss': detail['loss'],
            'epoch': detail['epoch'],
            'warm_up': detail['warm_up'],
        })

    warm_up = best['warm_up']
    warm_iteration = len(train_dataloader) * args.warm_epoch
    scheduler = lr_scheduler.StepLR(optim, step_size=args.step_size, gamma=0.1)
    for epoch in range(best['epoch'] + 1, args.epoch):
        print(f'\n----- epoch {epoch} -----')
        util.set_seed(epoch)
        if epoch < args.warm_epoch:
            warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
            run_cls_nn(args, 'train', model, train_dataloader, warm_up=warm_up, criterion=criterion, optim=optim,
                   apex=args.apex)
        else:
            run_cls_nn(args, 'train', model, train_dataloader, criterion=criterion, optim=optim,
                   apex=args.apex)
        with torch.no_grad():
            val = run_cls_nn(args, 'valid', model, valid_dataloader, criterion=criterion)

        detail = {
            'loss': val['loss'],
            'epoch': epoch,
            'warm_up': warm_up,
        }
        if val['loss'] <= best['loss']:
            best.update(detail)
        if epoch % 5 ==0:
            if args.apex:
                util.save_model(model, optim, 'cls_model_apex', detail, amp=amp)
            else:
                util.save_model(model, optim, 'cls_model', detail)
        print('[best] ep:%d loss:%.4f' % (best['epoch'], best['loss']))

        scheduler.step()


def run_cls_nn(args, mode, model, loader, warm_up=None, criterion=None, optim=None, apex=False):
    if mode in ['train']:
        model.train()
    elif mode in ['valid', 'test']:
        model.eval()
    else:
        raise

    t1 = time.time()

    losses = []
    ids_all = []
    outputs_all = []
    targets_all = []

    for i, (inputs, targets, filenames) in enumerate(loader):
        if args.use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)

        if mode in ['train', 'valid']:
            loss = criterion(outputs, targets)

            with torch.no_grad():
                losses.append(loss.item())

        if mode in ['train']:
            if warm_up:
                loss *= warm_up
            if apex:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optim.step()
            optim.zero_grad()

        with torch.no_grad():
            # outputs = np.argmax(outputs.cpu().numpy(), axis=1)
            outputs = np.where(outputs.cpu().numpy() > 0.5, 1, 0)

            outputs_all.extend(outputs)
            ids_all.extend(filenames)
            targets = targets.cpu().numpy()
            targets_all.extend(targets)

            accuracy = np.sum(outputs == targets) / len(targets)
            # accuracy = 0
            # for id in range(len(outputs)):
            #     accuracy += np.array_equal(outputs[id], targets[id])
            # accuracy /= len(targets)
            # outputs_all['char_id'].extend(out_char_ids.cpu().numpy())

        elapsed = int(time.time() - t1)
        eta = int(elapsed / (i + 1) * (len(loader) - (i + 1)))

        progress = f'\r[{mode}] {i + 1}/{len(loader)} {elapsed}(s) eta:{eta}(s) loss:{(np.sum(losses) / (i + 1)):.6f}  accuacy: {accuracy:.6f}\n'
        print(progress, end='')
        sys.stdout.flush()

    result = {
        'ids': np.array(ids_all),
        'outputs': np.array(outputs_all),
        'loss': np.sum(losses) / (i + 1)
    }
    return result


def run_nn(args, mode, model, loader, warm_up=None, criterion=None, optim=None, apex=False):
    if mode in ['train']:
        model.train()
    elif mode in ['valid', 'test']:
        model.eval()
    else:
        raise

    t1 = time.time()

    losses = [[], []]
    ids_all = []
    outputs_all = {'heatmap': [], 'paf_map': [], 'char_id': [], 'id': [], 'image_name': []}

    for i, (inputs, heatmaps, paf_maps, char_ids, ids, image_names) in enumerate(loader):
        save_mat = {}
        if args.use_gpu:
            inputs = inputs.cuda()
            heatmaps = heatmaps.cuda()
            paf_maps = paf_maps.cuda()
            char_ids = char_ids.cuda()

        out_paf_maps, out_heatmaps = model(inputs)

        if mode in ['train', 'valid']:
            loss_1 = criterion(out_heatmaps, heatmaps)
            loss_2 = criterion(out_paf_maps, paf_maps)
            loss = loss_1 + loss_2
            with torch.no_grad():
                losses[0].append(loss_1.item())
                losses[1].append(loss_2.item())

        if mode in ['train']:
            if warm_up:
                loss *= warm_up
            if apex:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optim.step()
            optim.zero_grad()

        with torch.no_grad():
            outputs_all['heatmap'].extend(out_heatmaps.cpu().numpy())
            outputs_all['paf_map'].extend(out_paf_maps.cpu().numpy())
            outputs_all['id'].extend(ids)
            # outputs_all['char_id'].extend(out_char_ids.cpu().numpy())
            ids_all.extend(ids)

        elapsed = int(time.time() - t1)
        eta = int(elapsed / (i + 1) * (len(loader) - (i + 1)))
        progress = f'\r[{mode}] {i + 1}/{len(loader)} {elapsed}(s) eta:{eta}(s) heatmap loss:{(np.sum(losses[0]) / (i + 1)):.6f} paf_map loss:{(np.sum(losses[1]) / (i + 1)):.6f} \n'
        print(progress, end='')
        sys.stdout.flush()

    result = {
        'ids': np.array(ids_all),
        'outputs': np.array(outputs_all),
        'loss': np.sum(losses) / (i + 1)
    }
    return result


def run_refinement_nn(args, mode, model, loader, warm_up=None, criterion=None, optim=None, apex=False):
    if mode in ['train']:
        model.train()
    elif mode in ['valid', 'test']:
        model.eval()
    else:
        raise

    t1 = time.time()

    losses = []
    ids_all = []
    outputs_all = {'heatmap': [], 'paf_map': [], 'char_id': [], 'id': [], 'image_name': []}

    for i, (inputs, heatmaps, paf_maps, char_ids, ids, image_names) in enumerate(loader):
        save_mat = {}
        if args.use_gpu:
            inputs = inputs.cuda()
            heatmaps = heatmaps.cuda()
            paf_maps = paf_maps.cuda()
            char_ids = char_ids.cuda()

        outputs = model(inputs)

        if mode in ['train', 'valid']:
            loss_1 = criterion(outputs[0], heatmaps)
            loss_2 = criterion(outputs[1], paf_maps)
            loss_3 = criterion(outputs[2], heatmaps)
            loss_4 = criterion(outputs[2], heatmaps)
            loss = loss_1 + loss_2 + loss_3 + loss_4
            with torch.no_grad():
                losses.append(loss.item())

        if mode in ['train']:
            if warm_up:
                loss *= warm_up
            if apex:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optim.step()
            optim.zero_grad()

        with torch.no_grad():
            output_heatmap = (outputs[0].cpu().numpy() + outputs[2].cpu().numpy())/2
            output_paf_map = (outputs[1].cpu().numpy() + outputs[3].cpu().numpy())/2
            outputs_all['heatmap'].extend(output_heatmap)
            outputs_all['paf_map'].extend(output_paf_map)
            outputs_all['id'].extend(ids)
            # outputs_all['char_id'].extend(out_char_ids.cpu().numpy())
            ids_all.extend(ids)

        elapsed = int(time.time() - t1)
        eta = int(elapsed / (i + 1) * (len(loader) - (i + 1)))
        progress = f'\r[{mode}] {i + 1}/{len(loader)} {elapsed}(s) eta:{eta}(s) heatmap loss:{(np.sum(losses) / (i + 1)):.6f} \n'
        print(progress, end='')
        sys.stdout.flush()

    result = {
        'ids': np.array(ids_all),
        'outputs': np.array(outputs_all),
        'loss': np.sum(losses) / (i + 1)
    }
    return result


def train_openpose(args, model, train_dataloader, valid_dataloader):
    writer = SummaryWriter()
    params = model.parameters()
    optim = torch.optim.SGD([{'params': params, 'lr': args.lr}], weight_decay=5e-4, momentum=0.9, nesterov=True)
    # optim = torch.optim.AdamW(params)
    if args.apex:
        amp.initialize(model, optim, opt_level='O1')

    criterion = Loss()

    best = {
        'loss': float('inf'),
        'epoch': -1,
        'warm_up': 0.1,
    }

    if args.snapshot:
        if args.apex:
            detail = util.load_model(args.snapshot, model, optim=optim, amp=amp)
        else:
            detail = util.load_model(args.snapshot, model, optim=optim)
        best.update({
            'loss': detail['loss'],
            'epoch': detail['epoch'],
            'warm_up': detail['warm_up'],
        })

    warm_up = best['warm_up']
    warm_iteration = len(train_dataloader) * args.warm_epoch
    scheduler = lr_scheduler.StepLR(optim, step_size=args.step_size, gamma=0.1)
    for epoch in range(best['epoch'] + 1, args.epoch):
        print(f'\n----- epoch {epoch} -----')
        util.set_seed(epoch)
        if epoch < args.warm_epoch:
            warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
            run_openpose(args, 'train', model, train_dataloader, warm_up=warm_up, criterion=criterion, optim=optim,
                   apex=args.apex, writer=writer, epoch_id=epoch)
        else:
            run_openpose(args, 'train', model, train_dataloader, criterion=criterion, optim=optim,
                   apex=args.apex, writer=writer, epoch_id=epoch)
        with torch.no_grad():
            val = run_openpose(args, 'valid', model, valid_dataloader, criterion=criterion, writer=writer, epoch_id=epoch)

        detail = {
            'loss': val['loss'],
            'epoch': epoch,
            'warm_up': warm_up,
        }
        if val['loss'] <= best['loss']:
            best.update(detail)
        if (epoch+1) % 1 ==0:
            if args.apex:
                util.save_model(model, optim, 'openpose_model_apex', detail, amp=amp)
            else:
                util.save_model(model, optim, 'openpose_model', detail)
        print('[best] ep:%d loss:%.4f' % (best['epoch'], best['loss']))

        scheduler.step()


def run_openpose(args, mode, model, loader, warm_up=None, criterion=None, optim=None, apex=False, writer=None, epoch_id=None):
    if mode in ['train']:
        model.train()
    elif mode in ['valid', 'test']:
        model.eval()
    else:
        raise

    t1 = time.time()

    losses = []
    ids_all = []
    outputs_all = {'heatmap': [], 'paf_map': [], 'char_id': [], 'id': [], 'image_name': []}

    num_dataloader = len(loader)
    for i, (data_dict) in enumerate(loader):
        if args.use_gpu:
            data_dict['img'] = data_dict['img'].cuda()
            data_dict['heatmap'] = data_dict['heatmap'].cuda()
            data_dict['vecmap'] = data_dict['vecmap'].cuda()
            data_dict['maskmap'] = data_dict['maskmap'].cuda()

        outputs = model(data_dict)
        if mode in ['train', 'valid']:
            loss = criterion(outputs)['loss']
            with torch.no_grad():
                losses.append(loss.item())

            n_step = epoch_id * num_dataloader + i
            if mode in ['train']:
                writer.add_scalar('Loss/train', loss.item(), n_step)
            else:
                writer.add_scalar('Loss/valid', loss.item(), n_step)

        if mode in ['train']:
            if warm_up:
                loss *= warm_up
            if apex:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optim.step()
            optim.zero_grad()

        # with torch.no_grad():
        #     output_heatmap = (outputs[0].cpu().numpy() + outputs[2].cpu().numpy())/2
        #     output_paf_map = (outputs[1].cpu().numpy() + outputs[3].cpu().numpy())/2
        #     outputs_all['heatmap'].extend(output_heatmap)
        #     outputs_all['paf_map'].extend(output_paf_map)
        #     outputs_all['id'].extend(ids)
        #     outputs_all['char_id'].extend(out_char_ids.cpu().numpy())
        #     ids_all.extend(ids)

        elapsed = int(time.time() - t1)
        eta = int(elapsed / (i + 1) * (len(loader) - (i + 1)))
        progress = f'\r[{mode}] {i + 1}/{len(loader)} {elapsed}(s) eta:{eta}(s) total loss:{(np.sum(losses) / (i + 1)):.3f} \n'
        print(progress, end='')
        sys.stdout.flush()

    result = {
        'ids': np.array(ids_all),
        'outputs': None,
        'loss': np.sum(losses) / (i + 1)
    }
    return result


def main():
    args = get_args()

    with open('data/images_info.json', 'r', encoding='utf8') as fb:
        images_info = json.load(fb)
    with open('data/strokes-detail.json', 'r', encoding='utf8') as fb:
        strokes_info = json.load(fb)
    with open('data/char-strokes.json', 'r', encoding='utf8') as fb:
        char2strokes = json.load(fb)
    with open('data/data.json', 'r', encoding='utf8') as fb:
        annotations = json.load(fb)

    if args.model == 'openpose':
        stride = 8
        if args.mode == 'train':
            model = OpenPose('train', 'resnet50', pretrained=False, snapshot='/home/zcy/CASIA-Classification/output/model_99999.pt')
            # model = OpenPose('train', 'resnet50', pretrained=False,
            #                  snapshot='checkpoints/model_99999.pt')
            # model = SimpleOpenPose('train', 'resnet34', pretrained=False)
            # model = SimpleOpenPose('train', 'cnn', pretrained=False)

            def normal_init(m, mean, std):
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    m.weight.data.normal_(mean, std)
                    if m.bias.data is not None:
                        m.bias.data.zero_()
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.weight.data.fill_(1)
                    if m.bias.data is not None:
                        m.bias.data.zero_()

            # model = SimpleNet('train', snapshot='/home/zcy/CASIA-Classification/output/model_99999.pt')
            # if args.snapshot:
            #     model.load_state_dict(torch.load(args.snapshot)['model'])
            # model.load_state_dict(torch.load('/home/zcy/char-structure/checkpoints/openpose_model_apex_ep4.pt')['model'])

            # for p in model.parameters():
            #     p.data.fill_(0)

            model.train()
            if args.use_gpu:
                torch.cuda.set_device(args.gpu)
                model.cuda()

            strokes_info = strokes_info['strokes']
            annotations = annotations['content']
            stroke_tool = Stroke(strokes_info)
            dataset = OriOpenPoseDataset(annotations, images_info, 'data/peanut', stroke_tool=stroke_tool, stride=stride, mode='train')
            len_dataset = len(dataset)
            print('loaded %d records' % len_dataset)

            def my_collate(batch):
                len_batch = len(batch)  # original batch length
                batch = list(filter(lambda x: x is not None, batch))  # filter out all the Nones
                if len_batch > len(
                        batch):  # if there are samples missing just use existing members, doesn't work if you reject every sample in a batch
                    diff = len_batch - len(batch)
                    batch = batch + batch[:diff]  # assume diff < len(batch)
                return torch.utils.data.dataloader.default_collate(batch)

            train_set, valid_set = torch.utils.data.random_split(dataset, [int(len_dataset * 0.95),
                                                                           len_dataset - int(len_dataset * 0.95)])
            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                      pin_memory=True,
                                      shuffle=True, drop_last=True, collate_fn=my_collate)
            valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                      pin_memory=True,
                                      shuffle=False, drop_last=True, collate_fn=my_collate)
            # batch = next(iter(train_loader))
            # img = batch['img'][0].numpy()
            # heatmap = batch['heatmap'][0][89].numpy()
            # vecmap = batch['vecmap'][0].numpy()
            #
            # mdict = {'heatmap': heatmap, 'img': img, 'vecmap': vecmap}
            # savemat('output/test.mat', mdict)
            # exit(0)
            train_openpose(args, model, train_loader, valid_loader)

        if args.mode == 'test':
            # model = SimpleOpenPose('test', 'cnn', pretrained=False)
            # model = SimpleNet('test', snapshot='/home/zcy/CASIA-Classification/output/model_99999.pt')
            model = OpenPose('test', 'resnet50', pretrained=False,
                             snapshot='checkpoints/model_99999.pt')
            model.eval()
            if args.use_gpu:
                torch.cuda.set_device(args.gpu)
                model.cuda()
            strokes_info = strokes_info['strokes']
            annotations = annotations['content']
            stroke_tool = Stroke(strokes_info)
            dataset = OriOpenPoseDataset(annotations, images_info, 'data/peanut', stroke_tool=stroke_tool, stride=stride,
                                      mode='test')
            len_dataset = len(dataset)
            print('loaded %d records' % len_dataset)
            test_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                         pin_memory=True, shuffle=False)

            with torch.no_grad():
                save_openpose_result(args, model, 'openpose', test_dataloader)

    if args.model == 'refinement':
        # model = RefinementNet(num_refinement_stages=1, num_heatmaps=90, num_pafs=126, num_channels=128, snapshot='checkpoints/model_99999.pt')
        model = RefinementNet(num_refinement_stages=1, num_heatmaps=90, num_pafs=126,
                              model='resnet')

        if args.use_gpu:
            torch.cuda.set_device(args.gpu)
            model.cuda()

        stride = 8
        if args.mode == 'train':
            strokes_info = strokes_info['strokes']
            annotations = annotations['content']

            stroke_tool = Stroke(strokes_info)
            dataset = PeanutDataset(annotations, images_info, 'data/peanut', stroke_tool=stroke_tool, stride=stride, mode='train')
            len_dataset = len(dataset)
            print('loaded %d records' % len_dataset)
            train_set, valid_set = torch.utils.data.random_split(dataset, [int(len_dataset * 0.8),
                                                                           len_dataset - int(len_dataset * 0.8)])
            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                      pin_memory=True,
                                      shuffle=True, drop_last=True)
            valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                      pin_memory=True,
                                      shuffle=False, drop_last=True)

            train_refinement_net(args, model, train_loader, valid_loader)

        if args.mode == 'test':
            strokes_info = strokes_info['strokes']
            annotations = annotations['content']

            stroke_tool = Stroke(strokes_info)
            dataset = PeanutDataset(annotations, images_info, 'data/peanut', stroke_tool=stroke_tool, mode='test', stride=stride)
            len_dataset = len(dataset)
            print('loaded %d records' % len_dataset)
            test_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                         pin_memory=True, shuffle=False)

            with torch.no_grad():
                save_test_result(args, model, 'refinement', test_dataloader)

    if args.model == 'cls':
        model = SimpleClassifier('train', num_classes=3752)
        if args.use_gpu:
            torch.cuda.set_device(args.gpu)
            model.cuda()
        if args.mode == 'train':
            def my_collate(batch):
                len_batch = len(batch)  # original batch length
                batch = list(filter(lambda x: x is not None, batch))  # filter out all the Nones
                if len_batch > len(
                        batch):  # if there are samples missing just use existing members, doesn't work if you reject every sample in a batch
                    diff = len_batch - len(batch)
                    batch = batch + batch[:diff]  # assume diff < len(batch)
                return torch.utils.data.dataloader.default_collate(batch)
            dataset = PeanutClsDataset(images_info, 'data/peanut')
            print(dataset.char_counter)
            len_dataset = len(dataset)
            train_set, valid_set = torch.utils.data.random_split(dataset, [int(len_dataset * 0.9),
                                                                           len_dataset - int(len_dataset * 0.9)])

            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                      pin_memory=True,
                                      shuffle=True, drop_last=True, collate_fn=my_collate)
            valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                      pin_memory=True,
                                      shuffle=False, drop_last=True, collate_fn=my_collate)
            train_cls_net(args, model, train_loader, valid_loader)

    if args.model == 'base':
        model = bodypose_model()
        if args.use_gpu:
            torch.cuda.set_device(args.gpu)
            model.cuda()
        if args.mode == 'train':
            strokes_info = strokes_info['strokes']
            annotations = annotations['content']

            stroke_tool = Stroke(strokes_info)
            dataset = PeanutDataset(annotations, images_info, 'data/peanut', stroke_tool=stroke_tool, mode='train')
            len_dataset = len(dataset)
            print('loaded %d records' % len_dataset)
            train_set, valid_set = torch.utils.data.random_split(dataset, [int(len_dataset * 0.8), len_dataset - int(len_dataset * 0.8)])
            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                                      shuffle=True, drop_last=True)
            valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                                      shuffle=False, drop_last=True)

            train(args, model, train_loader, valid_loader)

        elif args.mode == 'test':
            strokes_info = strokes_info['strokes']
            annotations = annotations['content']

            stroke_tool = Stroke(strokes_info)
            dataset = PeanutDataset(annotations, images_info, 'data/peanut', stroke_tool=stroke_tool, mode='test')
            len_dataset = len(dataset)
            print('loaded %d records' % len_dataset)
            test_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

            with torch.no_grad():
                save_test_result(args, model, 'base', test_dataloader)
        else:
            debug()


def debug():

    with open('data/images_info.json', 'r', encoding='utf8') as fb:
        temp_images_info = json.load(fb)
    with open('data/strokes-detail.json', 'r', encoding='utf8') as fb:
        strokes_info = json.load(fb)
    with open('data/char-strokes.json', 'r', encoding='utf8') as fb:
        char2strokes = json.load(fb)
    with open('data/data.json', 'r', encoding='utf8') as fb:
        annotations = json.load(fb)

    # calculate number of keypoints and connections
    # num_keypoints = 0
    # num_connections = 0
    # for stroke in strokes_info:
    #     num_keypoints += stroke['strokeOrderLength']
    #     num_connections += stroke['strokeOrderLength'] - 1
    # print(num_keypoints, num_connections)
    #
    # read image
    # images_info = images_info['images']
    # first_item = images_info[3]
    # image_id = first_item['id']

    dataset = PeanutClsDataset(temp_images_info, imgdir='data/peanut')

    # image_ten, char_id, filename = dataset[100]
    # print(char_id, filename)
    # plt.imshow(image_ten.permute(1, 2, 0))
    # plt.show()
    # exit(0)

    char_list = {}
    num_char = 0
    for record in temp_images_info:
        if record['cId'] not in char_list:
            char_list[record['cId']] = 1
            num_char += 1
    print(num_char)

    strokes_info = strokes_info['strokes']
    annotations = annotations['content']

    record = annotations[1]
    cnt_image_id = str(record['currentImageId'])
    dataset_id = str(record['dataSetId'])
    image_name = '%s-%s.png' % (dataset_id, cnt_image_id)
    char_id = int(record['charId'])
    stroke_annotations = record['result']
    images_info = {}

    for image_info in temp_images_info:
        images_info[str(image_info['fileName'])] = image_info

    image_info = images_info[image_name]
    # image_name = image_info['fileName']
    image_name = '1001-1.png'
    sample = {
        'image': Image.open('%s/%s' % ('data/peanut', image_name)).resize((256,256)),
        'image_name': image_name,
        'annotation': stroke_annotations,
    }

    stroke_tool = Stroke(strokes_info)
    dataset = PeanutDataset(annotations, temp_images_info, 'data/peanut', stroke_tool=stroke_tool)

    # heatmap = dataset._get_heatmap(sample)
    # plt.show()

    outputs = loadmat('output/1001-1.png_result.mat')
    heatmap = outputs['heatmap']

    heatmap = util.resize_heatmap(heatmap[5,:,:], 8)
    print(heatmap.shape)
    show_heatmap(sample['image'], heatmap)
    #
    #
    # resize image using bilinear interpolation
    # img = img.resize((256, 256), 2)
    # image_data = np.asarray(img)
    # print(image_data.shape)
    #
    # get heatmap and paf maps
    # stroke_tool = Stroke(strokes_info['strokes'])
    # peanut = PeanutDataset(annotations, stroke_tool, 4, 1, 1)
    # sample = {'image': img, 'image_id': image_id}
    # heatmap = peanut._generate_heatmap(sample)
    # paf_maps = peanut._get_paf_map(sample)
    #
    # resize and save paf map to matrix
    # print(paf_maps.shape)
    # h, w = paf_maps.shape[1], paf_maps.shape[2]
    # paf_maps = paf_maps.reshape((-1, 2, h, w))
    # resized_paf_map = util.resize_paf_map(paf_maps[4,:,:,:], 4)
    # print(resized_paf_map.shape)
    # paf_mat = {}
    # for id in range(paf_maps.shape[0]):
    #     paf_map = util.resize_paf_map(paf_maps[id, :, :, :], 4)
    #     paf_mat['x_%d' % id] = paf_map[0]
    #     paf_mat['y_%d' % id] = paf_map[1]
    # print(paf_mat)
    # savemat('paf_map.mat', paf_mat)
    #
    # show heatmap
    # temp = util.resize_heatmap(heatmap[-1], 4)
    # plt.imshow(temp)
    # plt.show()
    # show_heatmap(img, temp)



def show_heatmap(image, heatmap):
    def transparent_cmap(cmap, N=255):
        "Copy colormap and set alpha values"
        mycmap = cmap
        mycmap._init()
        mycmap._lut[:, -1] = np.linspace(0, 0.8, N + 4)
        return mycmap

    # Use base cmap to create transparent
    mycmap = transparent_cmap(plt.cm.Reds)
    plt.imshow(image)
    plt.contourf(heatmap, cmap=mycmap)
    plt.show()


def save_paf_to_mat(paf, name):
    paf_mat = {}
    for i in range(len(paf)):
        dx = paf[i, 0, :, :]
        dy = paf[i, 1, :, :]
        paf_mat['dx_%d' % i] = dx
        paf_mat['dy_%d' % i] = dy
    savemat('output/{0}_paf.mat'.format(name), paf_mat)


def show_keypoints(image, strokes):
    pprint(strokes)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(image)

    for stroke in strokes:
        stroke_id = stroke['id']
        keypoints = np.array(stroke['record'])
        plt.scatter(keypoints[:,0], keypoints[:,1], marker='o')
    plt.show()


if __name__ == '__main__':
    main()
