import os

import unet3D
from unet3D import UNet3D

from tqdm import tqdm
from pathlib import Path
import json
import pickle

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
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

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


import torch





directory = os.environ.get("MONAI_DATA_DIRECTORY")
#root_dir = tempfile.mkdtemp() if directory is None else directory
root_dir = Path("/storage/brno2/home/yauheni/Unet3D")



train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)



dir_path = Path("/storage/brno2/home/yauheni/data/imagesTr")
dir1_path = Path("/storage/brno2/home/yauheni/data/labelsTr")
dir_path.mkdir(parents=True, exist_ok=True)
dir1_path.mkdir(parents=True, exist_ok=True)




directory_path = Path('/storage/brno2/home/yauheni/data') 
file_path = directory_path / "dataset_0.json"  

# Создаем данные
data = {
    "description": "btcv yucheng",
"labels": {
    "0": "background",
    "1": "spleen",
    "2": "rkid",
    "3": "lkid",
    "4": "gall",
    "5": "eso",
    "6": "liver",
    "7": "sto",
    "8": "aorta",
    "9": "IVC",
    "10": "veins",
    "11": "pancreas",
    "12": "rad",
    "13": "lad"
},
"licence": "yt",
"modality": {
    "0": "CT"
},
"name": "btcv",
"numTest": 20,
"numTraining": 80,
"reference": "Vanderbilt University",
"release": "1.0 06/08/2015",
"tensorImageSize": "3D",
"test": [
    "imagesTs/img0061.nii.gz",
    "imagesTs/img0062.nii.gz",
    "imagesTs/img0063.nii.gz",
    "imagesTs/img0064.nii.gz",
    "imagesTs/img0065.nii.gz",
    "imagesTs/img0066.nii.gz",
    "imagesTs/img0067.nii.gz",
    "imagesTs/img0068.nii.gz",
    "imagesTs/img0069.nii.gz",
    "imagesTs/img0070.nii.gz",
    "imagesTs/img0071.nii.gz",
    "imagesTs/img0072.nii.gz",
    "imagesTs/img0073.nii.gz",
    "imagesTs/img0074.nii.gz",
    "imagesTs/img0075.nii.gz",
    "imagesTs/img0076.nii.gz",
    "imagesTs/img0077.nii.gz",
    "imagesTs/img0078.nii.gz",
    "imagesTs/img0079.nii.gz",
    "imagesTs/img0080.nii.gz"
],
"training": [
    {
        "image": "imagesTr/img0001.nii.gz",
        "label": "labelsTr/label0001.nii.gz"
    },
    {
        "image": "imagesTr/img0002.nii.gz",
        "label": "labelsTr/label0002.nii.gz"
    },
    {
        "image": "imagesTr/img0003.nii.gz",
        "label": "labelsTr/label0003.nii.gz"
    },
    {
        "image": "imagesTr/img0004.nii.gz",
        "label": "labelsTr/label0004.nii.gz"
    },
    {
        "image": "imagesTr/img0005.nii.gz",
        "label": "labelsTr/label0005.nii.gz"
    },
    {
        "image": "imagesTr/img0006.nii.gz",
        "label": "labelsTr/label0006.nii.gz"
    },
    {
        "image": "imagesTr/img0007.nii.gz",
        "label": "labelsTr/label0007.nii.gz"
    },
    {
        "image": "imagesTr/img0008.nii.gz",
        "label": "labelsTr/label0008.nii.gz"
    },
    {
        "image": "imagesTr/img0009.nii.gz",
        "label": "labelsTr/label0009.nii.gz"
    },
    {
        "image": "imagesTr/img0010.nii.gz",
        "label": "labelsTr/label0010.nii.gz"
    },
    {
        "image": "imagesTr/img0021.nii.gz",
        "label": "labelsTr/label0021.nii.gz"
    },
    {
        "image": "imagesTr/img0022.nii.gz",
        "label": "labelsTr/label0022.nii.gz"
    },
    {
        "image": "imagesTr/img0023.nii.gz",
        "label": "labelsTr/label0023.nii.gz"
    },
    {
        "image": "imagesTr/img0024.nii.gz",
        "label": "labelsTr/label0024.nii.gz"
    },
    {
        "image": "imagesTr/img0025.nii.gz",
        "label": "labelsTr/label0025.nii.gz"
    },
    {
        "image": "imagesTr/img0026.nii.gz",
        "label": "labelsTr/label0026.nii.gz"
    },
    {
        "image": "imagesTr/img0027.nii.gz",
        "label": "labelsTr/label0027.nii.gz"
    },
    {
        "image": "imagesTr/img0028.nii.gz",
        "label": "labelsTr/label0028.nii.gz"
    },
    {
        "image": "imagesTr/img0029.nii.gz",
        "label": "labelsTr/label0029.nii.gz"
    },
    {
        "image": "imagesTr/img0030.nii.gz",
        "label": "labelsTr/label0030.nii.gz"
    },
    {
        "image": "imagesTr/img0031.nii.gz",
        "label": "labelsTr/label0031.nii.gz"
    },
    {
        "image": "imagesTr/img0032.nii.gz",
        "label": "labelsTr/label0032.nii.gz"
    },
    {
        "image": "imagesTr/img0033.nii.gz",
        "label": "labelsTr/label0033.nii.gz"
    },
    {
        "image": "imagesTr/img0034.nii.gz",
        "label": "labelsTr/label0034.nii.gz"
    }
],
"validation": [
    {
        "image": "imagesTr/img0035.nii.gz",
        "label": "labelsTr/label0035.nii.gz"
    },
    {
        "image": "imagesTr/img0036.nii.gz",
        "label": "labelsTr/label0036.nii.gz"
    },
    {
        "image": "imagesTr/img0037.nii.gz",
        "label": "labelsTr/label0037.nii.gz"
    },
    {
        "image": "imagesTr/img0038.nii.gz",
        "label": "labelsTr/label0038.nii.gz"
    },
    {
        "image": "imagesTr/img0039.nii.gz",
        "label": "labelsTr/label0039.nii.gz"
    },
    {
        "image": "imagesTr/img0040.nii.gz",
        "label": "labelsTr/label0040.nii.gz"
    }
]
}

# Проверяем существует ли директория. Если нет - создаем ее
#directory_path.mkdir(parents=True, exist_ok=True)

# Записываем данные в файл
with file_path.open('w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)




data_dir = "/storage/brno2/home/yauheni/data/"
split_json = "dataset_0.json"

datasets = data_dir + split_json
datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")
train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=24,
    cache_rate=1.0,
    num_workers=8,
)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)





os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet3D(in_channels=1, out_channels=14).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)



def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model_for_trans_learning.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


max_iterations = 25000
eval_num = 500
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model_for_trans_learning.pth")))




file_path = '/storage/brno2/home/yauheni/Unet3D/epoch_loss_values_for_trans_learning.pkl'

# Проверка существования файла и запись данных
if not os.path.exists(file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(epoch_loss_values, file)
    print(f"File created and data saved: {file_path}")
else:
    with open(file_path, 'wb') as file:
        pickle.dump(epoch_loss_values, file)
    print(f"Data saved to existing file: {file_path}")


file_path = '/storage/brno2/home/yauheni/Unet3D/metric_values_for_trans_learning.pkl'

# Проверка существования файла и запись данных
if not os.path.exists(file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(metric_values, file)
    print(f"File created and data saved: {file_path}")
else:
    with open(file_path, 'wb') as file:
        pickle.dump(metric_values, file)
    print(f"Data saved to existing file: {file_path}")

print("The end")
