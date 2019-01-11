import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from detectionNet import ret_net


images = 0

net = ret_net(images, nettype=2)
