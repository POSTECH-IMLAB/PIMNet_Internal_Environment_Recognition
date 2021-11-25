import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork

from .networks import SSH, MobileNetV1


class ClassHead(nn.Conv2d):

    def __init__(self, in_channels=512, num_anchors=3):
        super().__init__(in_channels, num_anchors*2, kernel_size=1)
        self.num_anchors = num_anchors

    def forward(self, input):
        out = self._conv_forward(input, self.weight, self.bias)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.size(0), -1, 2)


class BboxHead(nn.Conv2d):
    def __init__(self, in_channels=512, num_anchors=3):
        super().__init__(in_channels, num_anchors*4, kernel_size=1)

    def forward(self, input):
        out = self._conv_forward(input, self.weight, self.bias)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.size(0), -1, 4)


class LandmarkHead(nn.Conv2d):
    def __init__(self, in_channels=512, num_anchors=3):
        super().__init__(in_channels, num_anchors*10, kernel_size=1)

    def forward(self, input):
        out = self._conv_forward(input, self.weight, self.bias)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.size(0), -1, 10)


class RetinaFace(nn.Module):

    def __init__(self, backbone, in_channel, out_channel, **kwargs):
        super().__init__()
        assert backbone in ("mobilenet0.25", "resnet50")
        if backbone == "mobilenet0.25":
            model = MobileNetV1()
            checkpoint = torch.load("./weights/mobilenet0.25_pretrain.tar", map_location="cpu")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:]  # remove module.
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)
            return_nodes={
                "stage1": "feat0",
                "stage2": "feat1",
                "stage3": "feat2",
            }
        else:
            import torchvision.models as models
            model = models.resnet50(pretrained=True)
            return_nodes={
                "layer2": "feat0",
                "layer3": "feat1",
                "layer4": "feat2",
            }

        self.body = create_feature_extractor(model, return_nodes=return_nodes)
        in_channels_stage2 = in_channel
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = out_channel
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
        out = self.body(inputs)

        # FPN
        out = self.fpn(out)

        # SSH
        feature0 = self.ssh1(out["feat0"])
        feature1 = self.ssh2(out["feat1"])
        feature2 = self.ssh3(out["feat2"])

        # features = [feature1, feature2, feature3]
        bbox_regressions = torch.cat([
            self.bbox_head[0](feature0),
            self.bbox_head[1](feature1),
            self.bbox_head[2](feature2),
        ], dim=1)

        classifications = torch.cat([
            self.class_head[0](feature0),
            self.class_head[1](feature1),
            self.class_head[2](feature2),
        ], dim=1)

        lm_regressions = torch.cat([
            self.landmark_head[0](feature0),
            self.landmark_head[1](feature1),
            self.landmark_head[2](feature2),
        ], dim=1)

        if not self.training:
            classifications = F.softmax(classifications, dim=-1)
        return bbox_regressions, classifications, lm_regressions
