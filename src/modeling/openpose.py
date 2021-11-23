import torch
import torch.nn as nn
from collections import OrderedDict

from src.modeling.utils.loss import BASE_LOSS_DICT
from torchvision import models


loss_dict = {
    "loss_type": "openpose_loss",
    "loss_weights": {
        "openpose_loss":{
            "heatmap_loss0": 1.0, "heatmap_loss1": 1.0, "heatmap_loss2": 1.0, "heatmap_loss3": 1.0, "heatmap_loss4": 1.0,
            "heatmap_loss5": 1.0, "paf_loss0": 1.0, "paf_loss1": 1.0, "paf_loss2": 1.0, "paf_loss3": 1.0, "paf_loss4": 1.0,
            "paf_loss5": 1.0,
        }
      },
    "params": {
        "mse_loss": {
            "reduction": "sum"
        }
    },
}


def normal_init(m, mean=1., std=0.):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


class SimpleClassifier(nn.Module):
    def __init__(self, phase, num_classes):
        super(SimpleClassifier, self).__init__()
        self.phase = phase
        assert self.phase in ['train', 'valid', 'test'], 'Invalid phase.'
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Linear(256, num_classes, bias=True)

    def forward(self, x):
        features = self.backbone(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        return self.classifier(features)


class SimpleNet(nn.Module):
    def __init__(self, phase, paf_out=122, heatmap_out=88, snapshot=None):
        super(SimpleNet, self).__init__()
        self.phase = phase
        assert self.phase in ['train', 'valid', 'test'], 'Invalid phase.'
        self.valid_loss_dict = loss_dict['loss_weights']['openpose_loss']

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
        backbone = models.resnet50(pretrained=False)
        backbone.fc = nn.Linear(2048, 7330)
        if snapshot:
            checkpoint = torch.load(snapshot)
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                if 'classifier.weight' in k:
                    new_state_dict['fc.weight'] = v
                elif 'classifier.bias' in k:
                    new_state_dict['fc.bias'] = v
                else:
                    name = k[9:]
                    new_state_dict[name] = v
            backbone.load_state_dict(new_state_dict)
        self.backbone = nn.Sequential(*list(backbone.children())[:-4])

        # freeze parameters in backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.paf_net = nn.Sequential(
            nn.Conv2d(512, paf_out, 3, 1, 1),
        )

        self.heatmap_net = nn.Sequential(
            nn.BatchNorm2d(paf_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(paf_out, heatmap_out, 3, 1, 1),
        )

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    normal_init(m, 0.85, 0.09)
            except:
                normal_init(block)

    def forward(self, data_dict):
        features = self.backbone(data_dict['img'])
        paf_out = self.paf_net(features)
        heatmap_out = self.heatmap_net(paf_out)
        paf_out = [paf_out]
        heatmap_out = [heatmap_out]
        out_dict = dict(paf=paf_out[-1], heatmap=heatmap_out[-1])
        if self.phase == 'test':
            return out_dict

        loss_dict = dict()
        for i in range(len(paf_out)):
            if 'paf_loss{}'.format(i) in self.valid_loss_dict:
                loss_dict['paf_loss{}'.format(i)] = dict(
                    params=[paf_out[i] * data_dict['maskmap'], data_dict['vecmap'] * data_dict['maskmap']],
                    type=torch.cuda.LongTensor([BASE_LOSS_DICT['mse_loss']]),
                    weight=torch.cuda.FloatTensor([self.valid_loss_dict['paf_loss{}'.format(i)]])
                )

        for i in range(len(heatmap_out)):
            if 'heatmap_loss{}'.format(i) in self.valid_loss_dict:
                loss_dict['heatmap_loss{}'.format(i)] = dict(
                    params=[heatmap_out[i] * data_dict['maskmap'], data_dict['heatmap'] * data_dict['maskmap']],
                    type=torch.cuda.LongTensor([BASE_LOSS_DICT['mse_loss']]),
                    weight=torch.cuda.FloatTensor([self.valid_loss_dict['heatmap_loss{}'.format(i)]])
                )

        return out_dict, loss_dict


class SimpleOpenPose(nn.Module):
    def __init__(self, phase, backbone, pretrained=False, paf_out=122, heatmap_out=88):
        super(SimpleOpenPose, self).__init__()
        self.phase = phase
        assert self.phase in ['train', 'valid', 'test'], 'Invalid phase.'

        if backbone == 'resnet50':
            stride = 8
            # stride=8, num_features = 512
            self.backbone = nn.Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-4])
            num_features = 512
        if backbone == 'resnet34':
            stride = 8
            # stride=8, num_features = 512
            self.backbone = nn.Sequential(*list(models.resnet34(pretrained=pretrained).children())[:-4])
            num_features = 128
        if backbone == 'resnet18':
            stride = 8
            # stride=8, num_features = 512
            self.backbone = nn.Sequential(*list(models.resnet34(pretrained=pretrained).children())[:-4])
            num_features = 128
        if backbone == 'cnn':
            stride = 8
            backbone = nn.ModuleList()
            conv1 = nn.Conv2d(3, 64, 3, 1, 1)
            conv2 = nn.Conv2d(64, 64, 3, 1, 1)
            backbone.append(conv1)
            backbone.append(nn.MaxPool2d(4))
            backbone.append(nn.ReLU())
            backbone.append(conv2)
            backbone.append(nn.MaxPool2d(2))
            backbone.append(nn.ReLU())
            self.backbone = nn.Sequential(*backbone)
            num_features = 64

        # self.upsample = nn.Upsample(scale_factor=stride//4)
        # FIXME: Add Cpm here
        self.pose_model = SimplePoseModel(paf_out=paf_out, heatmap_out=heatmap_out, in_channels=num_features)
        self.valid_loss_dict = loss_dict['loss_weights']['openpose_loss']

    def forward(self, data_dict):
        x = self.backbone(data_dict['img'])
        paf_out, heatmap_out = self.pose_model(x)
        out_dict = dict(paf=paf_out[-1], heatmap=heatmap_out[-1])
        if self.phase == 'test':
            return out_dict

        loss_dict = dict()
        for i in range(len(paf_out)):
            if 'paf_loss{}'.format(i) in self.valid_loss_dict:

                loss_dict['paf_loss{}'.format(i)] = dict(
                    params=[paf_out[i]*data_dict['maskmap'], data_dict['vecmap']*data_dict['maskmap']],
                    type=torch.cuda.LongTensor([BASE_LOSS_DICT['mse_loss']]),
                    weight=torch.cuda.FloatTensor([self.valid_loss_dict['paf_loss{}'.format(i)]])
                )

        for i in range(len(heatmap_out)):
            if 'heatmap_loss{}'.format(i) in self.valid_loss_dict:
                loss_dict['heatmap_loss{}'.format(i)] = dict(
                    params=[heatmap_out[i]*data_dict['maskmap'], data_dict['heatmap']*data_dict['maskmap']],
                    type=torch.cuda.LongTensor([BASE_LOSS_DICT['mse_loss']]),
                    weight=torch.cuda.FloatTensor([self.valid_loss_dict['heatmap_loss{}'.format(i)]])
                )

        return out_dict, loss_dict


class OpenPose(nn.Module):
    def __init__(self, phase, backbone, pretrained=False, paf_out=122, heatmap_out=88, snapshot=None):
        super(OpenPose, self).__init__()
        self.phase = phase
        assert self.phase in ['train', 'valid', 'test'], 'Invalid phase.'

        if backbone == 'resnet50':
            stride = 8
            # stride=8, num_features = 512
            backbone = models.resnet50(pretrained=False)
            backbone.fc = nn.Linear(2048, 7330)
            if snapshot:
                checkpoint = torch.load(snapshot)
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    if 'classifier.weight' in k:
                        new_state_dict['fc.weight'] = v
                    elif 'classifier.bias' in k:
                        new_state_dict['fc.bias'] = v
                    else:
                        name = k[9:]  # 去掉 `module.`
                        new_state_dict[name] = v
                backbone.load_state_dict(new_state_dict)
            self.backbone = nn.Sequential(*list(backbone.children())[:-4])
            num_features = 512
        if backbone == 'resnet34':
            stride = 8
            # stride=8, num_features = 512
            self.backbone = nn.Sequential(*list(models.resnet34(pretrained=pretrained).children())[:-4])
            num_features = 128
        if backbone == 'resnet18':
            stride = 4
            # stride=8, num_features = 512
            self.backbone = nn.Sequential(*list(models.resnet34(pretrained=pretrained).children())[:-5])
            num_features = 64

        # FIXME: Add Cpm here
        self.pose_model = PoseModel(paf_out=paf_out, heatmap_out=heatmap_out, in_channels=num_features)
        self.valid_loss_dict = loss_dict['loss_weights']['openpose_loss']

    def forward(self, data_dict):
        x = self.backbone(data_dict['img'])
        paf_out, heatmap_out = self.pose_model(x)
        out_dict = dict(paf=paf_out[-1], heatmap=heatmap_out[-1])
        if self.phase == 'test':
            return out_dict

        loss_dict = dict()
        for i in range(len(paf_out)):
            if 'paf_loss{}'.format(i) in self.valid_loss_dict:

                loss_dict['paf_loss{}'.format(i)] = dict(
                    params=[paf_out[i], data_dict['vecmap']],
                    type=torch.cuda.LongTensor([BASE_LOSS_DICT['mse_loss']]),
                    weight=torch.cuda.FloatTensor([self.valid_loss_dict['paf_loss{}'.format(i)]])
                )

        for i in range(len(heatmap_out)):
            if 'heatmap_loss{}'.format(i) in self.valid_loss_dict:
                loss_dict['heatmap_loss{}'.format(i)] = dict(
                    params=[heatmap_out[i], data_dict['heatmap']],
                    type=torch.cuda.LongTensor([BASE_LOSS_DICT['mse_loss']]),
                    weight=torch.cuda.FloatTensor([self.valid_loss_dict['heatmap_loss{}'.format(i)]])
                )

        # for i in range(len(paf_out)):
        #     if 'paf_loss{}'.format(i) in self.valid_loss_dict:
        #
        #         loss_dict['paf_loss{}'.format(i)] = dict(
        #             params=[paf_out[i]*data_dict['maskmap'], data_dict['vecmap']*data_dict['maskmap']],
        #             type=torch.cuda.LongTensor([BASE_LOSS_DICT['mse_loss']]),
        #             weight=torch.cuda.FloatTensor([self.valid_loss_dict['paf_loss{}'.format(i)]])
        #         )
        #
        # for i in range(len(heatmap_out)):
        #     if 'heatmap_loss{}'.format(i) in self.valid_loss_dict:
        #         loss_dict['heatmap_loss{}'.format(i)] = dict(
        #             params=[heatmap_out[i]*data_dict['maskmap'], data_dict['heatmap']*data_dict['maskmap']],
        #             type=torch.cuda.LongTensor([BASE_LOSS_DICT['mse_loss']]),
        #             weight=torch.cuda.FloatTensor([self.valid_loss_dict['heatmap_loss{}'.format(i)]])
        #         )

        return out_dict, loss_dict

    def load_dict(self, path):
        backbone = self.backbone
        model_dict = backbone.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        backbone.load_state_dict(model_dict)
        print('backbone weights loaded.')


class RefineOpenPose(nn.Module):
    def __init__(self, phase, backbone, pretrained=False, paf_out=122, heatmap_out=88, snapshot=None):
        super(RefineOpenPose, self).__init__()
        self.phase = phase
        assert self.phase in ['train', 'valid', 'test'], 'Invalid phase.'

        if backbone == 'resnet50':
            stride = 8
            # stride=8, num_features = 512
            backbone = models.resnet50(pretrained=False)
            backbone.fc = nn.Linear(2048, 7330)
            if snapshot:
                checkpoint = torch.load(snapshot)
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    if 'classifier.weight' in k:
                        new_state_dict['fc.weight'] = v
                    elif 'classifier.bias' in k:
                        new_state_dict['fc.bias'] = v
                    else:
                        name = k[9:]  # 去掉 `module.`
                        new_state_dict[name] = v
                backbone.load_state_dict(new_state_dict)
            self.backbone = nn.Sequential(*list(backbone.children())[:-4])
            num_features = 512
        if backbone == 'resnet34':
            stride = 8
            # stride=8, num_features = 512
            self.backbone = nn.Sequential(*list(models.resnet34(pretrained=pretrained).children())[:-4])
            num_features = 128
        if backbone == 'resnet18':
            stride = 4
            # stride=8, num_features = 512
            self.backbone = nn.Sequential(*list(models.resnet34(pretrained=pretrained).children())[:-5])
            num_features = 64

        # FIXME: Add Cpm here
        self.pose_model = RefinePoseModel(paf_out=paf_out, heatmap_out=heatmap_out, in_channels=num_features)
        self.valid_loss_dict = loss_dict['loss_weights']['openpose_loss']

    def forward(self, data_dict):
        x = self.backbone(data_dict['img'])
        std_heatmap = data_dict['std_heatmap']
        std_paf = data_dict['std_paf']
        if torch.cuda.is_available():
            std_heatmap = data_dict['std_heatmap'].cuda()
            std_paf = data_dict['std_paf'].cuda()
        paf_out, heatmap_out = self.pose_model(x, std_heatmap=std_heatmap, std_paf=std_paf)
        out_dict = dict(paf=paf_out[-1], heatmap=heatmap_out[-1])
        if self.phase == 'test':
            return out_dict

        loss_dict = dict()
        for i in range(len(paf_out)):
            if 'paf_loss{}'.format(i) in self.valid_loss_dict:

                loss_dict['paf_loss{}'.format(i)] = dict(
                    params=[paf_out[i], data_dict['vecmap']],
                    type=torch.cuda.LongTensor([BASE_LOSS_DICT['mse_loss']]),
                    weight=torch.cuda.FloatTensor([self.valid_loss_dict['paf_loss{}'.format(i)]])
                )

        for i in range(len(heatmap_out)):
            if 'heatmap_loss{}'.format(i) in self.valid_loss_dict:
                loss_dict['heatmap_loss{}'.format(i)] = dict(
                    params=[heatmap_out[i], data_dict['heatmap']],
                    type=torch.cuda.LongTensor([BASE_LOSS_DICT['mse_loss']]),
                    weight=torch.cuda.FloatTensor([self.valid_loss_dict['heatmap_loss{}'.format(i)]])
                )
        # for i in range(len(paf_out)):
        #     if 'paf_loss{}'.format(i) in self.valid_loss_dict:
        #
        #         loss_dict['paf_loss{}'.format(i)] = dict(
        #             params=[paf_out[i]*data_dict['maskmap'], data_dict['vecmap']*data_dict['maskmap']],
        #             type=torch.cuda.LongTensor([BASE_LOSS_DICT['mse_loss']]),
        #             weight=torch.cuda.FloatTensor([self.valid_loss_dict['paf_loss{}'.format(i)]])
        #         )
        #
        # for i in range(len(heatmap_out)):
        #     if 'heatmap_loss{}'.format(i) in self.valid_loss_dict:
        #         loss_dict['heatmap_loss{}'.format(i)] = dict(
        #             params=[heatmap_out[i]*data_dict['maskmap'], data_dict['heatmap']*data_dict['maskmap']],
        #             type=torch.cuda.LongTensor([BASE_LOSS_DICT['mse_loss']]),
        #             weight=torch.cuda.FloatTensor([self.valid_loss_dict['heatmap_loss{}'.format(i)]])
        #         )

        return out_dict, loss_dict


class SimplePoseModel(nn.Module):
    def __init__(self, paf_out=122, heatmap_out=88, in_channels=512):
        super(SimplePoseModel, self).__init__()

        self.in_channels = in_channels
        model_dict = self._get_model_dict(paf_out, heatmap_out, in_channels)
        self.model0 = model_dict['block_0']
        self.model1_1 = model_dict['block1_1']
        self.model2_1 = model_dict['block2_1']

        self.model1_2 = model_dict['block1_2']
        self.model2_2 = model_dict['block2_2']

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    @staticmethod
    def _make_layers(layer_dict):
        layers = []

        for i in range(len(layer_dict) - 1):
            layer = layer_dict[i]
            for k in layer:
                v = layer[k]
                if 'pool' in k:
                    layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]

        layer = list(layer_dict[-1].keys())
        k = layer[0]
        v = layer_dict[-1][k]

        conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
        layers += [conv2d]

        return nn.Sequential(*layers)

    @staticmethod
    def _get_model_dict(paf_out, heatmap_out, in_channels):

        blocks = {}

        block_0 = [{'conv4_3_CPM': [in_channels, 64, 3, 1, 1]}, {'conv4_4_CPM': [64, 32, 3, 1, 1]}]

        blocks['block1_1'] = [{'conv5_1_CPM_L1': [32, 32, 3, 1, 1]}, {'conv5_2_CPM_L1': [32, 32, 3, 1, 1]},
                              {'conv5_3_CPM_L1': [32, 32, 3, 1, 1]}, {'conv5_4_CPM_L1': [32, 64, 1, 1, 0]},
                              {'conv5_5_CPM_L1': [64, paf_out, 1, 1, 0]}]

        blocks['block1_2'] = [{'conv5_1_CPM_L2': [32, 32, 3, 1, 1]}, {'conv5_2_CPM_L2': [32, 32, 3, 1, 1]},
                              {'conv5_3_CPM_L2': [32, 32, 3, 1, 1]}, {'conv5_4_CPM_L2': [32, 64, 1, 1, 0]},
                              {'conv5_5_CPM_L2': [64, heatmap_out, 1, 1, 0]}]

        for i in range(2, 3):
            blocks['block%d_1' % i] = [{'Mconv1_stage%d_L1' % i: [32 + paf_out + heatmap_out, 32, 7, 1, 3]},
                                       {'Mconv2_stage%d_L1' % i: [32, 32, 7, 1, 3]},
                                       {'Mconv3_stage%d_L1' % i: [32, 32, 7, 1, 3]},
                                       {'Mconv4_stage%d_L1' % i: [32, 32, 7, 1, 3]},
                                       {'Mconv5_stage%d_L1' % i: [32, 32, 7, 1, 3]},
                                       {'Mconv6_stage%d_L1' % i: [32, 32, 1, 1, 0]},
                                       {'Mconv7_stage%d_L1' % i: [32, paf_out, 1, 1, 0]}]
            blocks['block%d_2' % i] = [{'Mconv1_stage%d_L2' % i: [32 + paf_out + heatmap_out, 32, 7, 1, 3]},
                                       {'Mconv2_stage%d_L2' % i: [32, 32, 7, 1, 3]},
                                       {'Mconv3_stage%d_L2' % i: [32, 32, 7, 1, 3]},
                                       {'Mconv4_stage%d_L2' % i: [32, 32, 7, 1, 3]},
                                       {'Mconv5_stage%d_L2' % i: [32, 32, 7, 1, 3]},
                                       {'Mconv6_stage%d_L2' % i: [32, 32, 1, 1, 0]},
                                       {'Mconv7_stage%d_L2' % i: [32, heatmap_out, 1, 1, 0]}]

        layers = []
        for block in block_0:
            for key in block:
                v = block[key]
                if 'pool' in key:
                    layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]

        models = {
            'block_0': nn.Sequential(*layers)
        }

        for k in blocks:
            v = blocks[k]
            models[k] = PoseModel._make_layers(v)

        return models

    def forward(self, x):
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)

        paf_out = [out1_1, out2_1]
        heatmap_out = [out1_2, out2_2]
        return paf_out, heatmap_out


class PoseModel(nn.Module):
    def __init__(self, paf_out=122, heatmap_out=88, in_channels=512):
        super(PoseModel, self).__init__()

        self.in_channels = in_channels
        model_dict = self._get_model_dict(paf_out, heatmap_out, in_channels)
        self.model0 = model_dict['block_0']
        self.model1_1 = model_dict['block1_1']
        self.model2_1 = model_dict['block2_1']
        self.model3_1 = model_dict['block3_1']
        self.model4_1 = model_dict['block4_1']
        self.model5_1 = model_dict['block5_1']
        self.model6_1 = model_dict['block6_1']

        self.model1_2 = model_dict['block1_2']
        self.model2_2 = model_dict['block2_2']
        self.model3_2 = model_dict['block3_2']
        self.model4_2 = model_dict['block4_2']
        self.model5_2 = model_dict['block5_2']
        self.model6_2 = model_dict['block6_2']

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    @staticmethod
    def _make_layers(layer_dict):
        layers = []

        for i in range(len(layer_dict) - 1):
            layer = layer_dict[i]
            for k in layer:
                v = layer[k]
                if 'pool' in k:
                    layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]

        layer = list(layer_dict[-1].keys())
        k = layer[0]
        v = layer_dict[-1][k]

        conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
        layers += [conv2d]

        return nn.Sequential(*layers)

    @staticmethod
    def _get_model_dict(paf_out, heatmap_out, in_channels):

        blocks = {}

        block_0 = [{'conv4_3_CPM': [in_channels, 256, 3, 1, 1]}, {'conv4_4_CPM': [256, 128, 3, 1, 1]}]

        blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]}, {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                              {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]}, {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                              {'conv5_5_CPM_L1': [512, paf_out, 1, 1, 0]}]

        blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]}, {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                              {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]}, {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                              {'conv5_5_CPM_L2': [512, heatmap_out, 1, 1, 0]}]

        for i in range(2, 7):
            blocks['block%d_1' % i] = [{'Mconv1_stage%d_L1' % i: [128 + paf_out + heatmap_out, 128, 7, 1, 3]},
                                       {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
                                       {'Mconv7_stage%d_L1' % i: [128, paf_out, 1, 1, 0]}]
            blocks['block%d_2' % i] = [{'Mconv1_stage%d_L2' % i: [128 + paf_out + heatmap_out, 128, 7, 1, 3]},
                                       {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
                                       {'Mconv7_stage%d_L2' % i: [128, heatmap_out, 1, 1, 0]}]

        layers = []
        for block in block_0:
            for key in block:
                v = block[key]
                if 'pool' in key:
                    layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]

        models = {
            'block_0': nn.Sequential(*layers)
        }

        for k in blocks:
            v = blocks[k]
            models[k] = PoseModel._make_layers(v)

        return models

    def forward(self, x):
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        paf_out = [out1_1, out2_1, out3_1, out4_1, out5_1, out6_1]
        heatmap_out = [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2]
        return paf_out, heatmap_out


class RefinePoseModel(nn.Module):
    def __init__(self, paf_out=122, heatmap_out=88, in_channels=512):
        super(RefinePoseModel, self).__init__()

        self.in_channels = in_channels
        model_dict = self._get_model_dict(paf_out, heatmap_out, in_channels)
        self.model0 = model_dict['block_0']
        self.model1_1 = model_dict['block1_1']
        self.model2_1 = model_dict['block2_1']
        self.model3_1 = model_dict['block3_1']
        self.model4_1 = model_dict['block4_1']
        self.model5_1 = model_dict['block5_1']
        self.model6_1 = model_dict['block6_1']

        self.model1_2 = model_dict['block1_2']
        self.model2_2 = model_dict['block2_2']
        self.model3_2 = model_dict['block3_2']
        self.model4_2 = model_dict['block4_2']
        self.model5_2 = model_dict['block5_2']
        self.model6_2 = model_dict['block6_2']

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    @staticmethod
    def _make_layers(layer_dict):
        layers = []

        for i in range(len(layer_dict) - 1):
            layer = layer_dict[i]
            for k in layer:
                v = layer[k]
                if 'pool' in k:
                    layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]

        layer = list(layer_dict[-1].keys())
        k = layer[0]
        v = layer_dict[-1][k]

        conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
        layers += [conv2d]

        return nn.Sequential(*layers)

    @staticmethod
    def _get_model_dict(paf_out, heatmap_out, in_channels):

        blocks = {}

        block_0 = [{'conv4_3_CPM': [in_channels, 256, 3, 1, 1]}, {'conv4_4_CPM': [256, 128, 3, 1, 1]}]

        blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]}, {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                              {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]}, {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                              {'conv5_5_CPM_L1': [512, paf_out, 1, 1, 0]}]

        blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]}, {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                              {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]}, {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                              {'conv5_5_CPM_L2': [512, heatmap_out, 1, 1, 0]}]

        for i in range(2, 7):
            blocks['block%d_1' % i] = [{'Mconv1_stage%d_L1' % i: [128 + paf_out + heatmap_out, 128, 7, 1, 3]},
                                       {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
                                       {'Mconv7_stage%d_L1' % i: [128, paf_out, 1, 1, 0]}]
            blocks['block%d_2' % i] = [{'Mconv1_stage%d_L2' % i: [128 + paf_out + heatmap_out, 128, 7, 1, 3]},
                                       {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                       {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
                                       {'Mconv7_stage%d_L2' % i: [128, heatmap_out, 1, 1, 0]}]

        layers = []
        for block in block_0:
            for key in block:
                v = block[key]
                if 'pool' in key:
                    layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]

        models = {
            'block_0': nn.Sequential(*layers)
        }

        for k in blocks:
            v = blocks[k]
            models[k] = PoseModel._make_layers(v)

        return models

    def forward(self, x, std_heatmap, std_paf):
        out1 = self.model0(x)

        # model1_1: (128+paf_out) -> paf_out
        # out1_1 = self.model1_1(torch.cat([out1, std_paf], dim=1))
        out1_1 = self.model1_1(out1)
        # model2_1: (128+heatmap_out) -> heatmap_out
        # out1_2 = self.model1_2(torch.cat([out1, std_heatmap], dim=1))
        out1_2 = self.model1_2(out1)
        out1_2 = out1_2 * std_heatmap
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        paf_out = [out1_1, out2_1, out3_1, out4_1, out5_1, out6_1]
        heatmap_out = [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2]
        return paf_out, heatmap_out


if __name__ == "__main__":
    model = SimpleOpenPose('train', 'resnet18', False)
    input = {
        'img': torch.randn(16, 3, 256, 256),
        'heatmap': torch.randn(16, 90, 64, 64),
        'vecmap': torch.randn(16, 126, 64, 64),
        'maskmap': torch.randn(16, 1, 64, 64)
    }
    output = model(input)
    print(output)
