from torch.utils.data import Dataset
from torchvision.transforms import *
import numpy as np
import glob
from PIL import Image
import random
from tqdm import tqdm

class VCIP_Training(Dataset):
    """
    Randomized image patchifier with a buffer
    """
    def __init__(self, samples_per_img=20, patch_size=512, buffer_size=1000) -> None:
        super().__init__()
        dataset_glob = "/home/xyhang/dataset/VCIP2023_challenge/Train/*/*.png"
        self.samples_per_img = samples_per_img
        self.buffer_size = buffer_size
        self.buffer = []
        self.buffer_items = 0
        self.image_list = glob.glob(dataset_glob)
        self.len_image_list = len(self.image_list)
        self.patch_size = patch_size
        
        print("Initializing training buffer ...")
        self._fill_buffer()
        print("Buffer initialized ...")
    
    def __len__(self):
        return self.len_image_list*self.samples_per_img

    def _fill_buffer(self):
        if self.buffer_items > 0:
            return
        
        while(self.buffer_items < self.buffer_size):
            index = np.random.randint(self.len_image_list)
            img = Image.open(self.image_list[index])
            img = ToTensor()(img)

            # random crop N patches on the image
            cropper = RandomCrop(self.patch_size)
            for _ in range(self.samples_per_img):
                patch = cropper(img)
                self.buffer.append(patch)
                self.buffer_items += 1
                if self.buffer_items == self.buffer_size:
                    break
        random.shuffle(self.buffer)

    def __getitem__(self, _):
        self._fill_buffer()
        self.buffer_items -= 1
        return self.buffer.pop()
    
class VCIP_Validation(Dataset):
    """
    Randomized image patchifier with a buffer
    If stable: return image centers
    """
    def __init__(self, patch_size=512) -> None:
        super().__init__()
        dataset_glob = "/home/xyhang/dataset/VCIP2023_challenge/Validation/*/*.png"
        self.buffer = []
        self.image_list = glob.glob(dataset_glob)
        self.len_image_list = len(self.image_list)
        self.patch_size = patch_size
        
        print("Initializing buffer ...")
        self._fill_buffer()
        print("Buffer initialized ...")
    
    def __len__(self):
        return self.len_image_list

    def _fill_buffer(self):
        for i in tqdm(range(self.len_image_list), "Filling stable buffer"):
            img = Image.open(self.image_list[i])
            img = ToTensor()(img)

            # Crop out the image centers
            cropper = CenterCrop(self.patch_size)
            patch = cropper(img)
            self.buffer.append(patch)
            self.buffer_items += 1

    def __getitem__(self, idx):
        return self.buffer[idx]