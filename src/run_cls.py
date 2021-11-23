# %% [code]
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import lr_scheduler
import torchvision.models as models

from torchvision import transforms, datasets
from PIL import Image

# %% [code]
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0)),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.85,0.85,0.85], [0.09,0.09,0.09])
])
test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.85,0.85,0.85], [0.09,0.09,0.09])
])


# %% [code]
def check_valid(image_path):
    is_valid = True
    try:
        img = Image.open(image_path)
        img.close()
    except:
        is_valid = False
    return is_valid


train_path = '/home/zcy/datasets/CASIA/CASIA-HWDB_Train/Train'
test_path = '/home/zcy/datasets/CASIA/CASIA-HWDB_Test/Test'
train_dataset = datasets.ImageFolder(train_path, train_transforms, is_valid_file=check_valid)
test_dataset = datasets.ImageFolder(test_path, test_transforms, is_valid_file=check_valid)

# %% [code]
batch_size = 32
num_workers = 2
num_classes = 7330

# %% [code]
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                               pin_memory=True, shuffle=True, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                              pin_memory=True, shuffle=False, drop_last=False)


# %% [code]
def train(model, train_dataloader, valid_dataloader, lr=3e-4, warm_epoch=0, step_size=10, num_epochs=30, eval_period=5):
    params = model.parameters()
    # optim = torch.optim.SGD([{'params': params, 'lr': lr}], weight_decay=5e-4, momentum=0.9, nesterov=True)
    optim = torch.optim.AdamW(params)
    # FIXME: Add mask if useful
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCEWithLogitsLoss()

    best = {
        'loss': float('inf'),
        'epoch': -1,
        'warm_up': 0.1,
        'accuracy': 0,
    }

    warm_up = best['warm_up']
    warm_iteration = len(train_dataloader) * warm_epoch
    scheduler = lr_scheduler.StepLR(optim, step_size=step_size, gamma=0.1)
    for epoch in range(best['epoch'] + 1, num_epochs):
        print(f'\n----- epoch {epoch} -----')
        torch.manual_seed(epoch)
        torch.cuda.manual_seed(epoch)
        if epoch < warm_epoch:
            warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
            run_cls_nn('train', model, train_dataloader, warm_up=warm_up, criterion=criterion, optim=optim)
        else:
            run_cls_nn('train', model, train_dataloader, criterion=criterion, optim=optim)
        if (epoch + 1) % eval_period == 0:
            with torch.no_grad():
                val = run_cls_nn('valid', model, valid_dataloader, criterion=criterion)
            detail = {
                'loss': val['loss'],
                'epoch': epoch,
                'warm_up': warm_up,
                'accuracy': val['accuracy'],
            }
            if val['accuracy'] <= best['accuracy']:
                best.update(detail)
                torch.save(model.state_dict(), ' /home/zcy/CASIA-Classification/output/model_best.pt')

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), '/home/zcy/CASIA-Classification/output/model_%d.pt' % epoch)
        print('[best] ep:%d loss:%.4f acc:%.4f' % (best['epoch'], best['loss'], best['accuracy']))

        scheduler.step()


# %% [code]
def run_cls_nn(mode, model, loader, warm_up=None, criterion=None, optim=None, use_gpu=True, save_steps=20000):
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

    for i, (inputs, targets) in enumerate(loader):
        if use_gpu:
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

            loss.backward()
            optim.step()
            optim.zero_grad()

        with torch.no_grad():
            outputs = np.argmax(outputs.cpu().numpy(), axis=1)
            # outputs = np.where(outputs.cpu().numpy() > 0.5, 1, 0)
            outputs_all.extend(outputs)
            targets = targets.cpu().numpy()
            targets_all.extend(targets)
            accuracy = np.sum(outputs == targets) / len(targets)
        #             print(targets)
        #             print(outputs)
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

        if (i + 1) % save_steps == 0:
            torch.save(model.state_dict(), '/home/zcy/CASIA-Classification/output/model_%d.pt' % i)

    result = {
        'outputs': np.array(outputs_all),
        'loss': np.sum(losses) / (i + 1),
        'accuracy': accuracy,
    }
    return result


# %% [code]
def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels,
                  bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_dw_no_bn(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels,
                  bias=False),
        nn.ELU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.ELU(inplace=True),
    )


class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class BackboneNet(nn.Module):
    def __init__(self, num_classes=7338, num_channels=128):
        super(BackboneNet, self).__init__()
        self.backbone = self.model = nn.Sequential(
            conv(1, 32, stride=2, bias=False),
            conv_dw(32, 64),
            conv_dw(64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=1),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)  # conv5_5
        )
        self.cpm = Cpm(512, num_channels)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.fc1 = nn.Linear(6272, 4096, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096, bias=True)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        out = self.backbone(x)
        out = self.cpm(out)

        out = self.relu(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        # print(x.shape)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out


class SimpleClassifier(nn.Module):
    def __init__(self, phase, num_classes):
        super(SimpleClassifier, self).__init__()
        self.phase = phase
        assert self.phase in ['train', 'valid', 'test'], 'Invalid phase.'
        # self.backbone = nn.Sequential(
        #     nn.Conv2d(3, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(64, 128, 3, 1, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(128, 256, 3, 1, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.backbone = nn.Sequential(*list(models.resnet34(pretrained=False).children())[:-4])
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(2048, num_classes, bias=True)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# %% [code] {"scrolled":true,"_kg_hide-output":false}
torch.cuda.set_device(3)
# model = BackboneNet(num_classes=num_classes)
model = SimpleClassifier('train', num_classes=num_classes)
model.cuda()

train(model, train_dataloader, test_dataloader, lr=0.5, eval_period=1)
