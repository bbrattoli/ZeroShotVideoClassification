############################# LIBRARIES ######################################
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

"""=================================================================================================================="""


def get_network(opt):
    """
    Selection function for available networks.
    """
    if 'r3d' in opt.network:
        network = models.video.r3d_18
    elif '2plus1d' in opt.network:
        network = models.video.r2plus1d_18
    elif 'c3d' in opt.network:
        return C3D(fixconvs=opt.fixconvs, nopretrained=opt.nopretrained)
    else:
        raise Exception('Network {} not available!'.format(opt.network))
    return ResNet18(network, fixconvs=opt.fixconvs, nopretrained=opt.nopretrained)


"""=================================================================================================================="""


class ResNet18(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """

    def __init__(self, network, fixconvs=False, nopretrained=True):
        super(ResNet18, self).__init__()
        self.model = network(pretrained=nopretrained)
        if fixconvs:
            for param in self.model.parameters():
                param.requires_grad = False

        self.regressor = nn.Linear(self.model.fc.in_features, 300)
        self.dropout = torch.nn.Dropout(p=0.05)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        # model.fc.weight.requires_grad = True
        # model.fc.bias.requires_grad = True

    def forward(self, x):
        bs, nc, ch, l, h, w = x.shape
        x = x.reshape(bs*nc, ch, l, h, w)
        x = self.model(x)
        x = x.view(bs*nc, -1)
        x = x.reshape(bs, nc, -1)
        x = torch.mean(x, 1)
        x = self.dropout(x)
        x = self.regressor(x)
        x = F.normalize(x)
        return x


"""=================================================================================================================="""

"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.

C3D code taken from: https://github.com/DavideA/c3d-pytorch
"""


class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self, fixconvs=False, nopretrained=True):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.10)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        if nopretrained:
            self.load_state_dict(torch.load('/workplace/c3d.pickle'))

        self.regressor = nn.Linear(4096, 300)

        if fixconvs:
            for model in [self.conv1, self.conv2,
                          self.conv3a, self.conv3b,
                          self.conv4a, self.conv4b,
                          self.conv5a, self.conv5b,
                          self.fc6]:
                for param in model.parameters():
                    param.requires_grad = False

    def forward(self, x):
        bs, nc, ch, l, h, w = x.shape
        x = x.reshape(bs*nc, ch, l, h, w)

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        # h = self.relu(self.fc7(h))
        # h = self.dropout(h)

        # logits = self.fc8(h)
        # probs = self.softmax(logits)

        h = h.reshape(bs, nc, -1)
        h = torch.mean(h, 1)
        h = h.reshape(bs, -1)

        h = self.regressor(h)
        h = torch.nn.functional.normalize(h, dim=-1)
        return h


