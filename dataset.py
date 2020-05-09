import glob
import random
import os
import re
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def resize(image, size):
    image = F.interpolate(
        image.unsqueeze(0), size=(size, size), mode="nearest"
    ).squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = Image.open(img_path)
        # Resize
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        img = torch.from_numpy(img)

        return img_path, img

    def __len__(self):
        return len(self.files)


class MixUpDataset(Dataset):
    def __init__(
        self,
        list_path,
        img_size=416,
        augment=True,
        multiscale=True,
        normalized_labels=True,
        preproc=None,
        beta_values=(1.5, 1.5),
    ):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.preproc = preproc
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self._mixup_args = beta_values

    def __getitem__(self, index):
        # ---------
        #  Image
        # ---------
        img1, box1 = self.read_img_and_annot(index)
        lambd = 1
        if self._mixup_args is not None:
            lambd = max(0, min(1, np.random.beta(*self._mixup_args)))

        if lambd >= 1:
            weights1 = np.ones((box1.shape[0], 1))
            box1 = np.hstack((box1, weights1))
            _, height, width = img1.shape
            if self.preproc is not None:
                img1, box1 = self.preproc(img1, box1, (self.img_size, self.img_size))
            unmix_img = torch.from_numpy(img1).type(torch.FloatTensor)
            padded_labels = torch.cat(
                (
                    torch.zeros(box1.shape[0], 1),
                    torch.from_numpy(box1[:, -2].reshape(-1, 1)).type(
                        torch.FloatTensor
                    ),
                    torch.from_numpy(box1[:, :4]).type(torch.FloatTensor),
                ),
                dim=1,
            )
            return unmix_img, padded_labels

        index2 = int(np.random.choice(np.delete(np.arange(len(self)), index)))
        img2, box2 = self.read_img_and_annot(index2)

        # mixup two images
        height = max(img1.shape[1], img2.shape[1])
        width = max(img1.shape[2], img2.shape[2])
        mix_img = np.zeros((3, height, width), dtype=np.float32)
        mix_img[:, : img1.shape[1], : img1.shape[2]] = img1.astype(np.float32) * lambd
        mix_img[:, : img2.shape[1], : img2.shape[2]] += img2.astype(np.float32) * (
            1.0 - lambd
        )
        # mix_img = mix_img.astype(np.uint8)

        y1 = np.hstack((box1, np.full((box1.shape[0], 1), lambd)))
        y2 = np.hstack((box2, np.full((box2.shape[0], 1), 1.0 - lambd)))
        mix_label = np.vstack((y1, y2))
        if self.preproc is not None:
            mix_img, padded_labels = self.preproc(
                mix_img, mix_label, (self.img_size, self.img_size)
            )

        mix_img = torch.from_numpy(mix_img).type(torch.FloatTensor)
        padded_labels = torch.cat(
            (
                torch.zeros(mix_label.shape[0], 1),
                torch.from_numpy(mix_label[:, -2].reshape(-1, 1)).type(
                    torch.FloatTensor
                ),
                torch.from_numpy(mix_label[:, :4]).type(torch.FloatTensor),
            ),
            dim=1,
        )

        return mix_img, padded_labels

    def read_img_and_annot(self, index):
        file_details = self.img_files[index % len(self.img_files)].rstrip()
        tmp_split = re.split(r"( \d)", file_details, maxsplit=1)
        if len(tmp_split) > 2:
            line = tmp_split[0], tmp_split[1] + tmp_split[2]
        else:
            line = tmp_split

        img = Image.open(line[0]).convert("RGB")
        line = line[1].split(" ")
        iw, ih = img.size
        h, w = self.img_size, self.img_size
        box = np.array(
            [np.array(list(map(int, box.split(",")))) for box in line[1:]],
            dtype=np.float32,
        )

        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        img = img.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(img, (dx, dy))
        img = np.array(new_image).reshape((3, h, w))

        # correct boxesa
        if len(box) > 0:
            #             np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            b_x = (box[:, 2] + box[:, 0]) * 0.5
            b_y = (box[:, 3] + box[:, 1]) * 0.5
            b_w = (box[:, 2] - box[:, 0]) * 1.0
            b_h = (box[:, 3] - box[:, 1]) * 1.0
            box[:, 0] = b_x / w  # *(w/iw)
            box[:, 1] = b_y / h  # *(h/ih)
            box[:, 2] = b_w / w  # *(w/iw)
            box[:, 3] = b_h / h  # *(w/iw)
        return img, box

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([img for img in imgs])
        self.batch_count += 1
        return imgs, targets

    def __len__(self):
        return len(self.img_files)
