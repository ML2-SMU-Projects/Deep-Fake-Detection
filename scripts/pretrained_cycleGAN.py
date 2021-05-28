"""
CycleGAN from
https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/master/p1ch2/3_cyclegan.ipynb
"""

import os

import requests
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

DATA_URL = 'https://raw.githubusercontent.com/deep-learning-with-pytorch/dlwpt-code/master/data/p1ch2/'
WEIGHTS_ENDPOINT = 'horse2zebra_0.4.0.pth'
HORSE_ENDPOINT = 'horse.jpg'


### get the pretrained weights and image

if not os.path.exists('./.cache/'):
    os.mkdir('./.cache/')
for endpoint in [HORSE_ENDPOINT, WEIGHTS_ENDPOINT]:
    if not os.path.exists('./cache/' + endpoint):
        r = requests.get(DATA_URL + endpoint)
        with open('./.cache/' + endpoint, 'wb') as f:
            f.write(r.content)



class ResNetBlock(nn.Module):
    """
    ResNet module
    """

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x) 
        return out


class ResNetGenerator(nn.Module):
    """
    Generator of ResNet Modules
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9): 

        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        # stack ResNet Modules
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input): 
        return self.model(input)

### Create generator

netG = ResNetGenerator()
model_weights = torch.load('./.cache/' + WEIGHTS_ENDPOINT)
netG.load_state_dict(model_weights)

### create preprocess pipeline

preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.ToTensor()])

# open the image

img = Image.open('./.cache/' + HORSE_ENDPOINT)

# preprocess the image

img_t = preprocess(img)                 # convert to tensor
batch_t = torch.unsqueeze(img_t, 0)        # convert single tensor to batch

# pass batch through network

batch_out = netG(batch_t)

# save output to image

out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)
out_img.save('./zebra.jpg')
