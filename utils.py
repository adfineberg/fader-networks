import pandas as pd
import numpy as np
import torch
from dataset import Dataset
from PIL import Image
from os.path import join
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale, Lambda
from torchvision.transforms import RandomHorizontalFlip


def plot_samples(x, x_hat, fpaths):
    x = x.data.cpu().numpy().transpose(0, 2, 3, 1)
    x_hat = x_hat.data.cpu().numpy().transpose(0, 2, 3, 1)

    for i in range(x.shape[0]):
        img_in  = 255 * ((x[i] + 1) / 2)
        img_out = 255 * ((x_hat[i] + 1) / 2)
        stacked = np.hstack((img_in, img_out))
        img = Image.fromarray(stacked.astype(np.uint8))
        img.save(fpaths[i])


def input_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        Scale(256),
        RandomHorizontalFlip(),
        ToTensor(),                   # [0, 255] --> [ 0., 1.]
        Lambda(lambda x: 2 * x - 1),  # [0., 1.] --> [-1., 1.]
    ])


def input_transform_augment(crop_size):
    return Compose([
        CenterCrop(crop_size),
        Scale(256),
        ToTensor(),                   # [0, 255] --> [ 0., 1.]
        Lambda(lambda x: 2 * x - 1),  # [0., 1.] --> [-1., 1.]
    ])


def target_transform():
    return Compose([
        Lambda(lambda x: torch.from_numpy((x + 1) / 2).float()),
    ])


# not sure if this is the best way
# transform 0 --> [0, 1], 1 --> [1, 0]
def target_transform_binary():
    def onehot_to_binary(x):
        z = np.empty(2 * x.shape[0], dtype=x.dtype)
        z[0::2] = (x + 1) / 2
        z[1::2] = 1 - (x + 1) / 2

        return torch.from_numpy(z).float()

    return Compose([
        Lambda(onehot_to_binary),
    ])


def create_partition():
    frame = pd.read_csv(join('data', 'list_attr_celeba.txt'), delim_whitespace=True, usecols=['Smiling'])
    shuffled_names = np.random.permutation(np.array(frame.index))
    shuffled_names[40000:] = shuffled_names[40000:] + ',0'
    shuffled_names[20000:40000] = shuffled_names[20000:40000] + ',1'
    shuffled_names[:20000] = shuffled_names[:20000] + ',2'
    np.savetxt('list_eval_partition.txt', shuffled_names, fmt="%s")


def split_train_val_test(data_dir):
    df = pd.read_csv(
        join(data_dir, 'list_eval_partition.txt'),
        header=None
    )
    filenames, labels = df.values[:, 0], df.values[:, 1]

    train_filenames = filenames[labels == 0]
    valid_filenames = filenames[labels == 1]
    test_filenames  = filenames[labels == 2]

    train_set = Dataset(
        data_dir, train_filenames, input_transform_augment(178),
        target_transform(), target_transform_binary()
    )
    valid_set = Dataset(
        data_dir, valid_filenames, input_transform(178),
        target_transform(), target_transform_binary()
    )
    test_set = Dataset(
        data_dir, test_filenames, input_transform(178),
        target_transform(), target_transform_binary()
    )

    return train_set, valid_set, test_set


if __name__ == '__main__':
    split_train_val_test('data')
