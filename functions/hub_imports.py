import os
import shutil
import warnings

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torchvision
from torchvision.io import read_image
from torchvision.models import ResNet50_Weights, resnet50
import torchvision.transforms as transforms
from torchray.attribution.grad_cam import grad_cam