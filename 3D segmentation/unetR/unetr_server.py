import numpy as np
import nibabel as nib
from pathlib import Path
import json 
from PIL import Image
from random import randint
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math
import torch.nn.functional as F
import copy
from monai.losses import DiceCELoss
from monai.networks.nets import UNETR
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
)
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
from tqdm import tqdm
import os
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from sklearn.model_selection import train_test_split
import pickle


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataloader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    def __len__(self):
        return len(self.dl)
    


masks_path = Path(r'/storage/brno2/home/yauheni/unetr3D/masks_for_all_organs')
data_path = Path(r'/storage/brno2/home/yauheni/workspase')

sorted_pictures = sorted(
        [d.name for d in data_path.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x)
    )

sorted_masks = sorted(
        [d.name for d in masks_path.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x)
    )



images_3d_full = []
for image_3d in sorted_pictures:
    images_path = Path(fr'/storage/brno2/home/yauheni/workspase/{image_3d}/data')
    images = []
    for image in images_path.iterdir():
        img = Image.open(image)
        img = img.convert('L')
        img = np.array(img)
        images.append(img)
    images_3d_full.append(np.array(images))
print("Shape of the first 3D image: ", images_3d_full[0].shape)



masks_3d_full = []
for mask_3d in sorted_masks:
    masks_path = Path(fr'/storage/brno2/home/yauheni/unetr3D/masks_for_all_organs/{mask_3d}')
    masks = []
    for mask in masks_path.iterdir():
        img = Image.open(mask)
        img = np.array(img)
        masks.append(img)
    masks_3d_full.append(np.array(masks))
print("Shape of the first 3D mask", masks_3d_full[0].shape)



class DataTrain(Dataset):

    def __init__(self, data, annotation):
        self.traininputtensor = torch.tensor(data, dtype=torch.float).unsqueeze(0)
        self.output = torch.tensor(annotation, dtype=torch.float)
    
    def __getitem__(self, index):
        input_image = self.traininputtensor[index] 
        output_label = self.output[index]  
        return input_image, output_label

    def __len__(self):
        return self.traininputtensor.size(0)


class DataTest(Dataset):

    def __init__(self, data, annotation):
        self.testinputtensor = torch.tensor(data, dtype=torch.float).unsqueeze(0)
        self.output = torch.tensor(annotation, dtype=torch.float)

    def __getitem__(self, index):
        input_image = self.testinputtensor[index]
        output_label = self.output[index] 
        return input_image, output_label

    def __len__(self):
        return self.testinputtensor.size(0)



def reshape_mask(mask, num_clases=8, depth=272):
    """We wanted to convert our 3d masks that have shape (1,D,H,W) to the shape (num_classes,D,H,W) """
    new_mask = np.zeros((num_clases, depth, 512, 512), dtype=np.uint8)

    for z in range(depth):  #iteration over 'z' axis 
        for y in range(512):    #iteration over 'y' axis
            for x in range(512):    #iteration over 'x' axis
                if mask[z,x,y] != 0:

                    value = mask[x,y]
                    trida = value/25
                    new_mask[trida, z, x, y] = 1

    return new_mask



def cut_data(data, z_shape=272):
    """Metacentrum haven't got enough memory to process one full image, that's why it will be cutted """
    cutted_data = []

    for i in range(0, data.shape[0], z_shape):
        new_data = np.array(data[i:i+z_shape, :, :])
        cutted_data.append(new_data)

    #for neural network we must have same dimension
    if cutted_data[-1].shape[0] < z_shape:
        cutted_data.pop(-1)

    return cutted_data



device = get_default_device()


model = UNETR(
    in_channels=1,
    out_channels=8,
    img_size=(272, 512, 512),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)



modelwise = UNETR(   #just_to_copy_weights
    in_channels=1,
    out_channels=14,
    img_size=(96, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)


modelwise.load_state_dict(torch.load(os.path.join(r"/storage/brno2/home/yauheni", "best_metric_model.pth")))
model_state_dict1 = modelwise.state_dict()
for name_dst, param_dst in model.named_parameters():
    if name_dst in modelwise.state_dict():
        param_src = model.state_dict()[name_dst]
        if param_src.size() == param_dst.size():
            param_dst.data.copy_(param_src.data)
        else:
            print(f"Skipping layer {name_dst} due to size mismatch")





trainX = []
trainY = []


cutted_5 = cut_data(images_3d_full[4])
cutted_14 = cut_data(images_3d_full[13])

cutted_mask_5 = cut_data(masks_3d_full[4])
cutted_mask_14 = cut_data(masks_3d_full[13])

for i in range(len(cutted_5)):
    trainX.append(cutted_5[i])
    trainY.append(reshape_mask(cutted_mask_5[i]))

for i in range(len(cutted_14)):
    trainX.append(cutted_14[i])
    trainY.append(reshape_mask(cutted_mask_14[i]))


X_for_mix = []
Y_for_mix = []
for i in range(len(images_3d_full)):

    if i != 4 and i != 13:

        cutted = cut_data(images_3d_full[i])
        cutted_mask = cut_data(masks_3d_full[i])

        X_for_mix.extend(cutted)
        Y_for_mix.extend(cutted_mask)


res_train_x, testX, res_train_y, testY = train_test_split(X_for_mix, Y_for_mix, test_size=0.33)
trainX.extend(res_train_x)
trainY.extend(res_train_y)



root_dir = r'/storage/brno2/home/yauheni/unetr3D'

data_train = DataTrain(np.array(trainX), np.array(trainY))
dataloader_train = DataLoader(dataset=data_train, batch_size=1, shuffle=True)

data_test = DataTrain(np.array(testX), np.array(testY))
dataloader_test = DataLoader(dataset=data_test, batch_size=1, shuffle=True)

dice_metric_with_bg = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)



loss_fn =  DiceCELoss(to_onehot_y=True, sigmoid=True)
lr = 1e-4
num_epochs = 2500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimazer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
history = []
history1 = []
acc_val = []
acc_test = []
acc_train = []
dice_metric_best = 0
number_of_epoch_best = 0
for epochs in range(num_epochs):
    with tqdm(total=17, desc=f'Epoch {epochs + 1}/{num_epochs}', unit='batch') as pbar:
        running_loss = 0
        val_loss = 0

        model.train()
        for i, data in enumerate(dataloader_train):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimazer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimazer.step()
            running_loss += loss.item()


            preds = (labels>0.5).float()
            output = (output>0.5).float()
            dice_metric_with_bg(y_pred=output, y=preds)
            dice_score = dice_metric_with_bg.aggregate().item()
            acc_train.append(dice_score)
            dice_metric_with_bg.reset()
            pbar.update(1)
        history.append(running_loss)
    correct = 0
    total = 0
    for j, data in enumerate(dataloader_test):
        with torch.no_grad():
            inputs_for_test, labels_for_test = data
            inputs_for_test = inputs_for_test.to(device)
            labels_for_test = labels_for_test.to(device)
            output_for_test = model(inputs_for_test)
            preds = (labels_for_test>0.5).float()
            output = (output_for_test>0.5).float()
            dice_metric_with_bg(y_pred=output, y=preds)
    dice_score = dice_metric_with_bg.aggregate().item()
    acc_val.append(dice_score)
    if dice_score>dice_metric_best:
        number_of_epoch_best = epochs
        dice_metric_best = dice_score
        torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model_pigs.pth"))
        print(
                "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_metric_best, dice_score)
        )
    else:
        print(
                "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_metric_best, dice_score
                    )
                )
        dice_metric_with_bg.reset()
print("Train history", history)
print("Accuracy train", acc_train)
print("Accuracy validation", acc_val)





file_path = r'/storage/brno2/home/yauheni/unetr3D/history_pigs.pkl'


if not os.path.exists(file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(history, file)
    print(f"File created and data saved: {file_path}")
else:
    with open(file_path, 'wb') as file:
        pickle.dump(history, file)
    print(f"Data saved to existing file: {file_path}")


file_path = r'/storage/brno2/home/yauheni/unetr3D/acc_train_pigs.pkl'


if not os.path.exists(file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(acc_train, file)
    print(f"File created and data saved: {file_path}")
else:
    with open(file_path, 'wb') as file:
        pickle.dump(acc_train, file)
    print(f"Data saved to existing file: {file_path}")



file_path = r'/storage/brno2/home/yauheni/unetr3D/acc_val.pkl'


if not os.path.exists(file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(acc_val, file)
    print(f"File created and data saved: {file_path}")
else:
    with open(file_path, 'wb') as file:
        pickle.dump(acc_val, file)
    print(f"Data saved to existing file: {file_path}")

print("The end")