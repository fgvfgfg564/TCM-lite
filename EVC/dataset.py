from torch.utils.data import Dataset
from torchvision.transforms import *
import numpy as np
import glob
from PIL import Image
import random

class VCIP_Training(Dataset):
    """
    Randomized image patchifier with a buffer
    """
    def __init__(self, samples_per_img=20, patch_size=512, buffer_size=1000, stable=False) -> None:
        super().__init__()
        dataset_glob = "/home/xyhang/dataset/VCIP2023_challenge/Train/*/*.png"
        self.image_list = glob.glob(dataset_glob)
        self.len_image_list = len(self.image_list)
        self.samples_per_img = samples_per_img
        self.patch_size = patch_size
        self.buffer_size = buffer_size
        self.buffer = []
        self.buffer_items = 0
        self.stable = stable

        if self.stable:
            random.shuffle(self.image_list)
            self.samples_per_img = 1
            self.len_image_list = self.buffer_size
            self.image_list = self.image_list[:self.buffer_size]
        
        self._fill_buffer()
    
    def __len__(self):
        return self.len_image_list*self.samples_per_img

    def _fill_buffer(self):
        if self.buffer_items > 0:
            return
        
        if self.stable:
            for i in range(self.buffer_size):
                img = Image.open(self.image_list[i])
                img = ToTensor()(img)

                # random crop N patches on the image
                cropper = CenterCrop(self.patch_size)
                patch = cropper(img)
                self.buffer.append(patch)
                self.buffer_items += 1
        else:
            while(self.buffer_items < self.buffer_size):
                index = np.random.randint(self.len_image_list)
                img = Image.open(self.image_list[index])
                img = ToTensor()(img)

                # random crop N patches on the image
                cropper = RandomCrop(self.patch_size)
                for i in range(self.samples_per_img):
                    patch = cropper(img)
                    self.buffer.append(patch)
                    self.buffer_items += 1
                    if self.buffer_items == self.buffer_size:
                        break
        
        random.shuffle(self.buffer)

    def __getitem__(self, idx):
        # Return a random image
        self._fill_buffer()
        if not self.stable:
            self.buffer_items -= 1
            return self.buffer.pop()
        else:
            return self.buffer[idx]