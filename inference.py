#!/usr/bin/env python

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from threading import Thread
import pickle
import argparse
import PIL

model=None
classes=None
input_resolution=None
predict=None
prob=None
stream = None
mean = None
std = None

def load_model(model_path):
    global model
    global SpectrumVariables
    global classes
    global input_resolution
    global mean
    global std

    model_data = torch.load(opt.model_path, map_location='cpu')
    input_resolution = model_data['resolution']
    classes = model_data['classes']
    mean = model_data['mean']
    std = model_data['std']
    found_model = False

    if model_data['modelType'] == 'densenet121':
        model = models.densenet121()
        model.classifier = nn.Linear(1024, len(classes))
        found_model = True
#         print('model is '+model_data['modelType'])
        if not found_model:
            print('could not find requested model:', model_data['modelType'])

    if model_data['modelType'] == 'resnet18':
        model = models.resnet18()
        model.fc = nn.Linear(512, len(classes))
        found_model = True
#         print('model is '+model_data['modelType'])
        if not found_model:
            print('could not find requested model:', model_data['modelType'])

    if model_data['modelType'] == 'resnet34':
        model = models.resnet34()
        model.fc = nn.Linear(512, len(classes))
        found_model = True
#         print('model is '+model_data['modelType'])
        if not found_model:
            print('could not find requested model:', model_data['modelType'])

    model.load_state_dict(model_data['model'])
    model.eval() # moved to here 25/2/2020
    if torch.cuda.is_available():
        device = torch.device("cuda")
#         print("device is cuda")
    else:
        device = torch.device("cpu")
#         print("device is cpu")
    model.to(device)
#     print('classes are: '+str(classes))


def infer_class(test_image):
    global predict
    global prob
    global classes
    global mean
    global std
    global model

#     input_resolution=SpectrumVariables["RESOLUTION"]

#     if model_data['inputType']=='cqt':
#         cqt = librosa.core.cqt(np.array(ringBuffer), sr=SpectrumVariables["SAMPLE_RATE"], n_bins=SpectrumVariables["N_BINS"], bins_per_octave=SpectrumVariables["BPO"], hop_length=SpectrumVariables["HOP_LENGTH"])
#         cqt_db = np.float32(librosa.amplitude_to_db(cqt, ref=np.max))

#         image=cqt_db[0:input_resolution,0:input_resolution]
#         image -= image.min() # ensure minimal value is 0.0
#         image /= image.max() # maximum value in image is 1.0
#         image*=256
#         image = image.astype(np.uint8)
#         color_image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
#     output_image = cv2.resize(color_image[:,-input_resolution:,:], (input_resolution, input_resolution))
#     cv2.imshow("rolling spectrogram", output_image)
#     cv2.waitKey(100)

#     image = Image.open(Path(test_image))


# try this next:

    image = PIL.Image.open(test_image)
    image = image.convert('RGB')
    image_tensor = transforms.Compose([
#         transforms.ToPILImage(), #?
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])(image)
    image_tensor = Variable(image_tensor, requires_grad=False)
    test_image = image_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        test_image = test_image.cuda()
    output = model(test_image)
    output = F.softmax(output, dim=1)
    prob, predict = torch.topk(output, len(classes))
    prob = str(list(prob[0].detach().cpu().numpy()))
    predict = str(predict[0].cpu().numpy())
    new_classes = str(list(classes))
    if predict == '[0 1]':
        print('computer says',classes[0],'!')
#         Image(opt.input_image)
    if predict == '[1 0]':
        print('computer says',classes[1],'!')
#         Image(opt.input_image)
    return

def run_inference(opt): #runtime is seconds
    print("having a think..... ")
    load_model(opt.model_path)
    infer_class(opt.test_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--test_image', type=str)
    opt = parser.parse_args()
    run_inference(opt)
