# -*- coding: utf-8 -*-
"""
Sripts to reproduce Lin's AtomSegNet.
Two sets of model weights of better performance are included based on evaluation: gaussianMask+ and superresolution
Optiuons for users: use cuda or not, resize the input image
(set as 1 to remain the same, set greater than 1 to upsampling and smaller than 1 to downsampling)

Note: the cuda is disabled by restricting 

"""

import os
from typing import Union, List
import numpy as np
from PIL import Image
from skimage import _raise_build_error
import torch
from torch.cuda import is_available
from torchvision.transforms import ToTensor
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.filters import sobel

from skimage.morphology import opening, disk, erosion
from skimage.segmentation import watershed

class Main():

    def __init__(self, inputdict):

        self.ori_image = None
        self.ori_content = None  # original image, PIL format
        self.model_output_content = None  # 2d array of model output
        self.result = None
        self.props = None
        self.coords = None
        self.inputdict = inputdict

    def LoadModel(self):

        def map8bit(data):
            return ((data - data.min()) / (data.max() - data.min()) * 255).astype('int8')

        self.ori_image = Image.fromarray(map8bit(self.imarray_original), 'L')
        if self.change_size == 1:
            self.ori_content = self.ori_image
        elif (self.change_size > 1):
            # Upsample using bicubic
            self.width, self.height = self.ori_image.size
            self.ori_content = self.ori_image.resize(
                (int(self.width * self.change_size), int(self.height * self.change_size)), Image.BICUBIC)
        else:
            # downsample using bilinear
            self.width, self.height = self.ori_image.size
            self.ori_content = self.ori_image.resize(
                (int(self.width * self.change_size), int(self.height * self.change_size)), Image.BILINEAR)

        self.width, self.height = self.ori_content.size

        self.result = np.zeros((self.height, self.width)) - 100
        temp_image = self.ori_content
        temp_result = self.load_model(self.modelname, temp_image)
        self.result = np.maximum(temp_result, self.result)
        self.result[self.result < 0] = 0
        self.model_output_content = (self.result - self.result.min()) / (self.result.max() - self.result.min())
        self.model_output_content = (self.model_output_content * 255 / np.max(self.model_output_content)).astype(
            'uint8')
        del temp_image
        del temp_result

    def CircleDetect(self):

        self.set_thre = 0  # 1 if set threshold
        self.thre = 100  # default threshold

        elevation_map = sobel(self.model_output_content)  # Sobel filter, a gradien filter for edge detection

        markers = np.zeros_like(self.model_output_content)
        if self.set_thre and self.thre:
            max_thre = int(self.thre) * 2.55
        else:
            max_thre = 100

        min_thre = 30
        markers[self.model_output_content < min_thre] = 1
        markers[self.model_output_content > max_thre] = 2

        seg_1 = watershed(elevation_map, markers)
        filled_regions = ndi.binary_fill_holes(seg_1 - 1)
        label_objects, nb_labels = ndi.label(filled_regions)
        self.props = regionprops(label_objects)

        del elevation_map
        del markers, seg_1, filled_regions, label_objects, nb_labels

        predictpos_y = []
        predictpos_x = []
        for p in self.props:
            c_y, c_x = p.centroid
            predictpos_y.append(c_y)
            predictpos_x.append(c_x)
        self.coords = np.asarray((predictpos_x, predictpos_y))

    def load_model(self, modelpath, data):

        if modelpath == "superresolution":
            model_path = "superresolution.pth"

            from NestedUNet import NestedUNet
            net = NestedUNet()

            # if cuda:
            #     net = net.cuda()
            # if cuda:
            #     net = torch.nn.DataParallel(net)
            #     net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
            # else:
            net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path, map_location=torch.device('cpu')).items()})

            transform = ToTensor()
            ori_tensor = transform(data)
            ori_tensor = torch.unsqueeze(ori_tensor, 0)

            padding_left = 0
            padding_right = 0
            padding_top = 0
            padding_bottom = 0
            ori_height = ori_tensor.size()[2]
            ori_width = ori_tensor.size()[3]
            use_padding = False
            if ori_width >= ori_height:
                padsize = ori_width
            else:
                padsize = ori_height
            if np.log2(padsize) >= 7.0:
                if (np.log2(padsize) - 7) % 1 > 0:
                    padsize = 2 ** (np.log2(padsize) // 1 + 1)
            else:
                padsize = 2 ** 7
            if ori_height < padsize:
                padding_top = int(padsize - ori_height) // 2
                padding_bottom = int(padsize - ori_height - padding_top)
                use_padding = True
            if ori_width < padsize:
                padding_left = int(padsize - ori_width) // 2
                padding_right = int(padsize - ori_width - padding_left)
                use_padding = True
            if use_padding:
                padding_transform = torch.nn.ConstantPad2d((padding_left, \
                                                            padding_right, \
                                                            padding_top, \
                                                            padding_bottom), 0)
                ori_tensor = padding_transform(ori_tensor)

            # if cuda:
            #     ori_tensor = ori_tensor.cuda()
            output = net(ori_tensor)

            # if cuda:
            #     result = (output.data).cpu().numpy()
            # else:
            result = (output.data).numpy()

            padsize = int(padsize)
            result = result[0, 0, padding_top:(padsize - padding_bottom),
                     padding_left:(padsize - padding_right)]

        elif modelpath == "gaussianMask+":
            from UNet import UNet
            model_path = "gaussianMask+.pth"

            use_padding = False
            unet = UNet()
            unet = unet.eval()

            # print("right before unet.cuda(), cuda bool is {}".format(cuda))
            # if cuda:
            #     unet = unet.cuda()

            # if not cuda:
            unet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
            # else:
            #     unet.load_state_dict(torch.load(model_path))

            print("after we load state dict, next(unet.parameters()).is_cuda is {}".format(
                next(unet.parameters()).is_cuda))

            # transform = ToTensor()
            ori_tensor = ToTensor()(data)
            # if cuda:
            #     # ori_tensor = Variable(ori_tensor.cuda())
            #     ori_tensor = ori_tensor.cuda()
            # else:
            # ori_tensor = Variable(ori_tensor)
            # ori_tensor = ori_tensor.cuda()
            ori_tensor = torch.unsqueeze(ori_tensor, 0)

            padding_left = 0
            padding_right = 0
            padding_top = 0
            padding_bottom = 0
            ori_height = ori_tensor.size()[2]
            ori_width = ori_tensor.size()[3]

            if ori_height % 4:
                padding_top = (4 - ori_height % 4) // 2
                padding_bottom = 4 - ori_height % 4 - padding_top
                use_padding = True
            if ori_width % 4:
                padding_left = (4 - ori_width % 4) // 2
                padding_right = 4 - ori_width % 4 - padding_left
                use_padding = True
            if use_padding:
                padding_transform = torch.nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom),
                                                           0)
                ori_tensor = padding_transform(ori_tensor)

            output = ori_tensor
            with torch.no_grad():
                output = unet(output)

            if use_padding:
                output = output[:, :, padding_top: (padding_top + ori_height), padding_left: (padding_left + ori_width)]

            result = (output.data).cpu().numpy()

            result = result[0, 0, :, :]
        else:
            raise Exception('Model not existing. Use superresolution or gaussianMask+')
        return result

    def run(self, *args):

        self.imarray_original = self.inputdict['image']
        self.modelname = self.inputdict['modelweights']
        self.change_size = self.inputdict['change_size']
        self.LoadModel()
        self.CircleDetect()
        return self.coords


def inference(dicts: Union[List[dict], dict]):
    """
    Run a model to predict atom column coordinates in the input STEM images
    modelweight: gaussianMask+ or superresolution
    change_size: set as 1 to remain the same, set greater than 1 to upsampling and smaller than 1 to downsampling

    Example:

    >>> # Make a dict of your inputs
    >>> # use a list of dicts if you want to run it though multiple images at one time
    >>> dict = {'image': a, 'modelweights':'superresolution','change_size': 1} 

    """
    results = []
    if isinstance(dicts, list):
      for i in range(len(dicts)):
        results.append(Main(dicts[i]).run())
      return results  
    elif isinstance(dicts, dict):
        results.append(Main(dicts).run())
        return results
    else: raise AssertionError("Input should be dict or list of dict")
    

