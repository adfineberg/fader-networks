import numpy as np
import torch.utils.data as data
from PIL import Image
from os.path import join
import pandas as pd


class Dataset(data.Dataset):
    def __init__(self, data_dir, filenames, input_transform,
                 target_transform, target_transform_binary, attr_names=['Smiling']):
        super(Dataset, self).__init__()
        image_dir = join(data_dir, 'img_align_celeba')
        filenames_lookup = set(filenames)

        frame = pd.read_csv(join(data_dir, 'list_attr_celeba.txt'), delim_whitespace=True, usecols=attr_names)
        attr_vals = frame.filter(items=filenames_lookup, axis=0).as_matrix()
        
        self.image_filenames = [join(image_dir, x) for x in filenames]
        # attr_vals = np.vstack(fname_to_attr[fname] for fname in filenames)
        self.attribute_names = np.array(attr_names)
        self.attribute_values = attr_vals
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.target_transform_binary = target_transform_binary

    def __getitem__(self, index):
        fp = self.image_filenames[index]
        x = self.input_transform(Image.open(fp))
        yb = self.target_transform_binary(self.attribute_values[index])
        yt = self.target_transform(self.attribute_values[index])

        return x, yb, yt, fp

    def __len__(self):
        return len(self.image_filenames)
