
import os
import re
import cv2
import json
import torch
import imgaug
import numpy as np
import matplotlib.pyplot as plt

from time import time
from imgaug import augmenters
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
from sklearn.model_selection import KFold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM_FINAL = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
TRANSFORM_RGB = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.52905, 0.52799, 0.52668], [0.32092, 0.32015, 0.32018])])
TRANSFORM_BW = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.53532, 0.53437, 0.53304], [0.31048, 0.30967, 0.30977])])


SAVE_PATH = "saves"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


class DS(Dataset):
    def __init__(self, bw_images: np.ndarray, bw_transform=None, rgb_images: np.ndarray = None, rgb_transform=None, mode: str = "train"):

        assert re.match(r"^train$", mode, re.IGNORECASE) or re.match(r"^valid$", mode, re.IGNORECASE) or re.match(r"^test$", mode, re.IGNORECASE), "Invalid Mode"
        
        self.mode = mode
        self.bw_transform = bw_transform
        self.bw_images = bw_images

        if re.match(r"^train$", mode, re.IGNORECASE) or re.match(r"^valid$", mode, re.IGNORECASE):
            self.rgb_transform = rgb_transform
            self.rgb_images = rgb_images
            
    def __len__(self):
        return self.bw_images.shape[0]

    def __getitem__(self, idx):
        if re.match(r"^train$", self.mode, re.IGNORECASE) or re.match(r"^valid$", self.mode, re.IGNORECASE):
            return self.bw_transform(self.bw_images[idx]), self.rgb_transform(self.rgb_images[idx])
        else:
            return self.bw_transform(self.bw_images[idx])


def breaker(num: int = 50, char: str = "*") -> None:
    print("\n" + num*char + "\n")


def load_data(path: str) -> tuple:
    assert "rgb_images.npy" in os.listdir(path) and "gray_images.npy" in os.listdir(path), "Please run python np_make.py"

    rgb_images = np.load(os.path.join(path, "rgb_images.npy"))
    bw_images = np.load(os.path.join(path, "gray_images.npy"))

    return rgb_images, bw_images


def get_augment(seed: int):
    imgaug.seed(seed)
    augment = augmenters.Sequential([
        augmenters.HorizontalFLip(p=0.15),
        augmenters.VerticalFLip(p=0.15),
        augmenters.Affine(scale=(0.5, 1.5), translate_percent=(-0.1, 0.1), rotate=(-45, 45)),
    ])
    return augment


def prepare_train_and_valid_dataloaders(path: str, mode: str, batch_size: int, seed: int, augment: bool=False):

    rgb_images, bw_images = load_data(path)

    for tr_idx, va_idx in KFold(n_splits=5, shuffle=True, random_state=seed).split(bw_images):
        tr_rgb_images, va_rgb_images, tr_bw_images, va_bw_images = rgb_images[tr_idx], rgb_images[va_idx], bw_images[tr_idx], bw_images[va_idx]
        break

    if augment:
        augmenter = get_augment(seed)
        tr_rgb_images = augmenter(images=tr_rgb_images)
        tr_bw_images = augmenter(images=tr_bw_images)

    
    if re.match(r"^full$", mode, re.IGNORECASE) or re.match(r"^semi$", mode, re.IGNORECASE):
        tr_data_setup = DS(tr_bw_images, TRANSFORM_BW, tr_rgb_images, TRANSFORM_RGB, "train")
        va_data_setup = DS(va_bw_images, TRANSFORM_BW, va_rgb_images, TRANSFORM_RGB, "valid")
    else:
        tr_data_setup = DS(tr_bw_images, TRANSFORM_FINAL, tr_rgb_images, TRANSFORM_FINAL, "train")
        va_data_setup = DS(va_bw_images, TRANSFORM_FINAL, va_rgb_images, TRANSFORM_FINAL, "train")

    dataloaders = {
        "train" : DL(tr_data_setup, batch_size=batch_size, shuffle=True, generator=torch.manual_seed(seed)),
        "valid" : DL(va_data_setup, batch_size=batch_size, shuffle=False)
    }

    return dataloaders


def save_graphs(L: list) -> None:
    TL, VL = [], []
    for i in range(len(L)):
        TL.append(L[i]["train"])
        VL.append(L[i]["valid"])
    x_Axis = np.arange(1, len(TL) + 1)
    plt.figure("Plots")
    plt.plot(x_Axis, TL, "r", label="Train")
    plt.plot(x_Axis, VL, "b", label="Valid")
    plt.legend()
    plt.grid()
    plt.title("Loss Graph")
    plt.savefig(os.path.join(SAVE_PATH, "Graphs.jpg"))
    plt.close("Plots")


def fit(model=None, optimizer=None, scheduler=None, epochs=None, early_stopping_patience=None, dataloaders=None, verbose=False):
    
    if verbose:
        breaker()
        print("Training ...")
        breaker()

    bestLoss = {"train" : np.inf, "valid" : np.inf}
    Losses = []
    name = "state.pt"

    start_time = time()
    for e in range(epochs):
        e_st = time()
        epochLoss = {"train" : 0.0, "valid" : 0.0}

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            lossPerPass = []

            for X, y in dataloaders[phase]:
                X, y = X.to(DEVICE), y.to(DEVICE)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    _, output = model(X)
                    loss = torch.nn.MSELoss()(output, y)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                lossPerPass.append(loss.item())
            epochLoss[phase] = np.mean(np.array(lossPerPass))
        Losses.append(epochLoss)

        if early_stopping_patience:
            if epochLoss["valid"] < bestLoss["valid"]:
                bestLoss = epochLoss
                BLE = e + 1
                torch.save({"model_state_dict": model.state_dict(),
                            "optim_state_dict": optimizer.state_dict()},
                           os.path.join(SAVE_PATH, name))
                early_stopping_step = 0
            else:
                early_stopping_step += 1
                if early_stopping_step > early_stopping_patience:
                    print("\nEarly Stopping at Epoch {}".format(e + 1))
                    break
        
        if epochLoss["valid"] < bestLoss["valid"]:
            bestLoss = epochLoss
            BLE = e + 1
            torch.save({"model_state_dict" : model.state_dict(),
                        "optim_state_dict" : optimizer.state_dict()},
                        os.path.join(SAVE_PATH, name))
        
        if scheduler:
            scheduler.step(epochLoss["valid"])
        
        if verbose:
            print("Epoch: {} | Train Loss: {:.5f} | Valid Loss: {:.5f} | Time: {:.2f} seconds".format(e+1, epochLoss["train"], epochLoss["valid"], time()-e_st))

    if verbose:                                           
        breaker()
        print(f"Best Validation Loss at Epoch {BLE}")
        breaker()
        print("Time Taken [{} Epochs] : {:.2f} minutes".format(len(Losses), (time()-start_time)/60))
        breaker()
        print("Training Completed")
        breaker()

    return Losses, BLE, name


def predict(model=None, mode: str = None, image_path: str = None, size: int = 320) -> np.ndarray:
    model.load_state_dict(torch.load(os.path.join(SAVE_PATH, "state.pt"), map_location=DEVICE)["model_state_dict"])
    model.eval()
    model.to(DEVICE)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w, _ = image.shape
    image = cv2.cvtColor(src=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY), code=cv2.COLOR_GRAY2RGB)
    image = cv2.resize(src=image, dsize=(size, size), interpolation=cv2.INTER_AREA)

    with torch.no_grad():
        if re.match(r"^full$", mode, re.IGNORECASE) or re.match(r"^semi$", mode, re.IGNORECASE):
            _, color_image = model(TRANSFORM_BW(image).to(DEVICE).unsqueeze(dim=0))
        else:
            _, color_image = model(TRANSFORM_FINAL(image).to(DEVICE).unsqueeze(dim=0))
    
    color_image = torch.sigmoid(color_image.squeeze())
    color_image = color_image.detach().cpu().numpy().transpose(1, 2, 0)
    color_image = np.clip((color_image * 255), 0, 255).astype("uint8")

    return cv2.resize(src=color_image, dsize=(w, h), interpolation=cv2.INTER_AREA)


def show(image: np.ndarray, title: bool = False) -> None:
    plt.figure()
    plt.imshow(image)
    plt.axis("off")
    if title:
        plt.title(title)
    figmanager = plt.get_current_fig_manager()
    figmanager.window.state("zoomed")
    plt.show()
