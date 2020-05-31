from __future__ import division
import re
import base64
import numpy as np
from PIL import Image
from io import BytesIO
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
from models import *
from utils.utils import *
from utils.datasets import *
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect,send_file
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import re
import numpy as np

#############import from PreNet

import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from networks import *
import time


def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image


def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="png")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")

def detect_image(image):
    """
    :param image : numpy RGB
    :return:
    """
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

    #####derain model ########
    derain_weight_root = 'F:/YOLO/venv/YOLOv3'          #use rain1400
    derain_model = PReNet(6, True)
    derain_model = derain_model.cuda()
    derain_model.load_state_dict(torch.load(os.path.join(derain_weight_root , 'net_latest.pth')))
    derain_model.eval()



    img = transforms.ToTensor()(Image.fromarray(image)).cuda()

    #####begin derain
    with torch.no_grad():
        derain_img = derain_model(img)
        derain_save_img = derain_img.cpu().numpy()*255
        derain_save_img = np.transpose(derain_save_img , (1,2,0))
    ####end derain
    # img = img.cpu()
    derain_img, _ = pad_to_square(derain_img, 0)
    # Resize
    derain_img = resize(derain_img, opt.img_size)
    input_imgs = Variable(derain_img.type(Tensor))
    input_imgs = torch.unsqueeze(input_imgs, dim=0)

    # Get detections
    with torch.no_grad():
        detections_list = model(input_imgs)
        detections_list = non_max_suppression(detections_list, opt.conf_thres, opt.nms_thres)

    del input_imgs
    # derain_img_array = derain_img.    # del derain_imgcpu().numpy()
    for detections in detections_list:
        img = derain_save_img.copy()
        if detections is not None:
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                label = str(classes[int(cls_pred)]) + str(cls_conf.item())
                cv2.putText(img, label, (x1+10, y1+10), font,1, (0, 0, 255), 1)
    return img

app = Flask(__name__)


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)
        if not isinstance(img , np.ndarray):
            img = np.array(img)
        img_return = detect_image(img)
        img_return = np_to_base64(img_return)
        # img1 img2 im3  110 220 330
        return jsonify(img_return)


        # return send_file(
        #     BytesIO(img_return),
        #     mimetype='image/png',
        #     as_attachment=True,
        #     attachment_filename='result.jpg'
        # )
        #return jsonify(result=result, probability=pred_proba)
    return None


if __name__ == '__main__':
    # http_server = WSGIServer(('192.168.31.25', 8000), app)
    # http_server.serve_forever()
    app.run(host='10.128.205.38', port=8000, debug=True)


