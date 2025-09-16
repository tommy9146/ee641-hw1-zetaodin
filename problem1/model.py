import torch
import torch.nn as nn

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class MultiScaleDetector(nn.Module):
    def __init__(self, num_classes=3, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Backbone: 4 blocks, 输出 3 个尺度
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),  # 224->112
        )
        self.block2 = conv_bn(64, 128, 2)   # 112->56 (Scale1)
        self.block3 = conv_bn(128, 256, 2)  # 56->28  (Scale2)
        self.block4 = conv_bn(256, 512, 2)  # 28->14  (Scale3)

        def head(ch):
            return nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, self.num_anchors*(5+self.num_classes), 1)
            )
        self.head1 = head(128)
        self.head2 = head(256)
        self.head3 = head(512)

    def forward(self, x):
        x = self.stem(x)
        s1 = self.block2(x)  # [B,128,56,56]
        s2 = self.block3(s1) # [B,256,28,28]
        s3 = self.block4(s2) # [B,512,14,14]
        p1 = self.head1(s1)
        p2 = self.head2(s2)
        p3 = self.head3(s3)
        return [p1, p2, p3]  # raw logits