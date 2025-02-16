import logging
import os

import torch
import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils_yolo.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils_yolo.torch_utils import select_device, smart_inference_mode

logger = logging.getLogger()

class yolov5(object):

    def __init__(self, save_path='', confidence_threshold=0.2, device="cuda:0", agnostic=False ):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = 0.5
        self.agnostic = agnostic
        self.half = False  # use FP16 half-precision inference
        self.dnn = False
        imgsz = (640, 640)
        assert os.path.exists(save_path), 'model pth is not exits'
        # print(sys.modules)
        device = select_device(device)
        self.net = DetectMultiBackend(save_path, device=device, dnn=self.dnn, fp16=self.half)
        stride, names, pt = self.net.stride, self.net.names, self.net.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        self.net.warmup(imgsz=(1 if pt or self.net.triton else 1, 3, *imgsz))

    def predict(self, img, ori_image):
        ori_size = ori_image[0].shape
        '''
        ori_size: h, w, _
        return: x, y, x, y, scores,  cls
        '''
        img = torch.from_numpy(img).to(self.net.device)
        img = img.half() if self.net.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        # Inference

        pred = self.net(img)

        # Apply NMS
        # print(pred)
        res_dets = []
        # pred = non_max_suppression(pred, self.confidence_threshold, self.nms_threshold, agnostic=self.agnostic)
        pred = non_max_suppression(pred, self.confidence_threshold, self.nms_threshold, gnostic=self.agnostic)
        index = 0
        for det in pred:
            ori_size = ori_image[index].shape
            index += 1
            if det is not None and len(det):
                # xyxy = det[det[:, 5] == 0]
                xyxy = det
                # if xyxy.shape[0] == 0:
                #     continue
                xyxy[:, :4] = scale_boxes(img.shape[2:], xyxy[:, :4], ori_size).round()
                res_dets.append(xyxy[:, :6].cpu())
                
            else:
                res_dets.append(torch.empty((0, 6)))
        # print(res_dets)
        
        return res_dets

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def scale_segments(img1_shape, segments, img0_shape, ratio_pad=None, normalize=False):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    segments[:, 0] -= pad[0]  # x padding
    segments[:, 1] -= pad[1]  # y padding
    segments /= gain
    clip_segments(segments, img0_shape)
    if normalize:
        segments[:, 0] /= img0_shape[1]  # width
        segments[:, 1] /= img0_shape[0]  # height
    return segments


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def clip_segments(segments, shape):
    # Clip segments (xy1,xy2,...) to image shape (height, width)
    if isinstance(segments, torch.Tensor):  # faster individually
        segments[:, 0].clamp_(0, shape[1])  # x
        segments[:, 1].clamp_(0, shape[0])  # y
    else:  # np.array (faster grouped)
        segments[:, 0] = segments[:, 0].clip(0, shape[1])  # x
        segments[:, 1] = segments[:, 1].clip(0, shape[0])  # y

