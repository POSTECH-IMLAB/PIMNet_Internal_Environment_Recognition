import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork

from models.net import FPN as FPN
from models.net import SSH as SSH
from models.net import MobileNetV1 as MobileNetV1


class ClassHead(nn.Module):

    def __init__(self, in_channels=512, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(
            in_channels, self.num_anchors*2, kernel_size=1, stride=1, padding=0
        )

    def forward(self, input):
        out = self.conv1x1(input)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.size(0), -1, 2)


class BboxHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1, padding=0
        )

    def forward(self, input):
        out = self.conv1x1(input)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.size(0), -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self,in_channels=512, num_anchors=3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels, num_anchors*10, kernel_size=1, stride=1, padding=0
        )

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.size(0), -1, 10)


class RetinaFace(nn.Module):

    def __init__(self, backbone, pretrain, return_layers, in_channel, out_channel, **kwargs):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super().__init__()
        if backbone == 'mobilenet0.25':
            model = MobileNetV1()
            if pretrain:
                checkpoint = torch.load("./weights/mobilenet0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                model.load_state_dict(new_state_dict)
            self.stage1 = model.stage1
            self.stage2 = model.stage2
            self.stage3 = model.stage3
        elif backbone == 'resnet50':
            import torchvision.models as models
            model = models.resnet50(pretrained=pretrain)
            self.stage1 = model.layer2
            self.stage2 = model.layer3
            self.stage3 = model.layer4
        else:
            raise NotImplementedError

        # self.body = _utils.IntermediateLayerGetter(model, return_layers)
        in_channels_stage2 = in_channel
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = out_channel
        # self.fpn = FPN(in_channels_list,out_channels)
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        fpn_num = len(in_channels_list)
        self.class_head = self._make_class_head(fpn_num=fpn_num, in_channels=out_channels)
        self.bbox_head = self._make_bbox_head(fpn_num=fpn_num, in_channels=out_channels)
        self.landmark_head = self._make_landmark_head(fpn_num=fpn_num, in_channels=out_channels)

    def _make_class_head(self, fpn_num=3, in_channels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(in_channels, anchor_num))
        return classhead
    
    def _make_bbox_head(self, fpn_num=3, in_channels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(in_channels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, in_channels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(in_channels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        # out = self.body(inputs)
        stage1 = self.stage1(inputs)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)

        # FPN
        feats = self.fpn({"1": stage1, "2": stage2, "3": stage3})

        # # SSH
        feature1 = self.ssh1(feats["1"])
        feature2 = self.ssh2(feats["2"])
        feature3 = self.ssh3(feats["3"])

        # features = [feature1, feature2, feature3]
        # bbox_regressions = torch.cat([self.bbox_head[i](feature) for i, feature in enumerate(features)], dim=1)
        # classifications = torch.cat([self.class_head[i](feature) for i, feature in enumerate(features)], dim=1)
        # ldm_regressions = torch.cat([self.landmark_head[i](feature) for i, feature in enumerate(features)], dim=1)
        bbox_regressions = torch.cat([
            self.bbox_head[0](feature1),
            self.bbox_head[1](feature2),
            self.bbox_head[2](feature3),
        ], dim=1)

        classifications = torch.cat([
            self.class_head[0](feature1),
            self.class_head[1](feature2),
            self.class_head[2](feature3),
        ], dim=1)

        lm_regressions = torch.cat([
            self.landmark_head[0](feature1),
            self.landmark_head[1](feature2),
            self.landmark_head[2](feature3),
        ], dim=1)

        if self.training:
            output = (bbox_regressions, classifications, lm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), lm_regressions)
        return output
