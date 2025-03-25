from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
from os import listdir

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class dehaze_train_dataset(Dataset):
    def __init__(self, input_dir, gt_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.filenames = [x for x in listdir(input_dir) if is_image_file(x)]

        self.input_dir = input_dir
        self.gt_dir = gt_dir

        self.file_len = len(self.filenames)


    def __getitem__(self, index):
        # print(self.input_dir + self.filenames[index])
        input = Image.open(self.input_dir + self.filenames[index]).convert('RGB')
        gt = Image.open(self.gt_dir + self.filenames[index]).convert('RGB')

        input = self.transform(input)
        gt = self.transform(gt)

        return input, gt
    

    def __len__(self):
        return self.file_len





