from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import cv2
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # delete
    # 1 image folder
    # 2batch_size
    # 3class_path
    # get image from cam

    # parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    # parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--v_cap", type=int, default=0,
                        help="number of video capture ，ls /dev/video to find all aviliable cam")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # os.makedirs("output", exist_ok=True)

    ##########################config model #####################

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    print("\nPerforming object detection:")
    print("-----open cam----- config paramters")
    cap = cv2.VideoCapture(opt.v_cap + cv2.CAP_DSHOW)
    cap.set(3, 512)  # set image size
    cap.set(4, 512)

    while True:
        r, f = cap.read()
        if r:
            f = f[:, :, ::-1]  # to rgb
            img = transforms.ToTensor()(Image.fromarray(f))
            img, _ = pad_to_square(img, 0)
            # Resize
            img = resize(img, opt.img_size)
            input_imgs = Variable(img.type(Tensor))
            input_imgs = torch.unsqueeze(input_imgs, dim=0)

        else:
            continue
        # Get detections
        with torch.no_grad():
            detections_list = model(input_imgs)
            detections_list = non_max_suppression(detections_list, opt.conf_thres, opt.nms_thres)

        del input_imgs

        for detections in detections_list:
            img = np.array(f[: , : , ::-1])
            if detections is not None:
                detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    label = str(classes[int(cls_pred)]) + str(cls_conf.item())
                    cv2.putText(img, label, (x1+10, y1+10), font,1, (0, 0, 255), 1)
                    #print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

        cv2.imshow("Video",img)
        c = cv2.waitKey(1)
        if c==27:
            break


    # close
    cap.release()
    cv2.destroyAllWindows()



