import numpy as np
import typing, types
# import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import os
import random
from torchvision.transforms import Compose as C
from torchvision import transforms as T


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.2):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def aug(p=0.3):
    return C([
                # AddGaussianNoise(),
                # normalize
                # T.Normalize(
                #     mean=[0.485, 0.456, 0.406],
                #     std=[0.229, 0.224, 0.225]
                # ),
                T.RandomHorizontalFlip(p=p),
                T.RandomVerticalFlip(p=p),
                T.ColorJitter(0.4, 0.4, 0.4, 0.2),
                T.RandomGrayscale(p=p),
                T.RandomPerspective(distortion_scale=0.50, p=p)
            ])



def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    index = np.argmax(memory_available[:-1])  # Skip the 7th card --- it is reserved for evaluation!!!

    return index  # Returns index of the gpu with the most memory available


class PartialMincDataset(Dataset):
    """
    There should be a separate instance for training, validation and testing
    """
    def __init__(self, file, category_dict, augmentation=False):
        self.label_name = list()
        self.img_names = list()
        self.labels = list()
        self.str2label = dict()
        self.invalid_files = list()
        self.get_item_func = self.get_item_runtime
        self.augmentation = augmentation
        label_dir = os.path.dirname(file)
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                split_line = line.split("/")
                if split_line[1] in category_dict:
                    self.label_name.append(split_line[1])
                    self.img_names.append(os.path.join(label_dir, "../", line).replace("\n", ""))
                    self.labels.append(category_dict[split_line[1]])
        print("Loaded partial dataset: "+file)

    def __getitem__(self, item):
        return self.get_item_func(item)

    def __len__(self):
        return len(self.labels)

    def get_item_runtime(self, item):
        with Image.open(self.img_names[item]) as im:
            # im = T.RandomVerticalFlip(p=0.9).forward(im)  # OK
            if self.augmentation:
                im = aug()(im)
            npim = np.array(im)
            img = npim.astype('f4')
            imgg = img.transpose((2, 0, 1)) / 255.0
            return imgg, self.labels[item]

    def get_item_safe(self, item):
        with Image.open(self.img_names[item]) as im:
            npim = np.array(im)
            # print(npim.shape)
            img = npim.astype('f4')
            imgg = np.zeros((3, 362, 362))
            try:
                imgg = img.transpose((2, 0, 1)) / 255.0
            except:
                self.invalid_files.append(self.img_names[item])
            # img = npim.astype('f4') / 255.0
            # print(im.shape)
            return imgg, self.labels[item]
        # print(self.img_names[item])

    def set_get_function(self, debug=True):
        if(debug):
            self.get_item_func = self.get_item_safe
        else:
            self.get_item_func = self.get_item_runtime

    def save_invalid_files(self, file_name="/home.nfs/hartvjir/invalid_files.txt"):
        with open(file_name, "a") as f:
            ret = ""
            for file in self.invalid_files:
                ret = ret + file + "\n"
            f.write(ret)


class MincDataset:
    def __init__(self, root_directory, dataset_id=1, augmentation=False, ignore_classes=()):
        assert 1 <= dataset_id <= 5
        self.root_directory = root_directory
        print("root_directory", root_directory)
        self.label_directory = os.path.join(root_directory, "labels")
        self.categories_path = os.path.join(root_directory, "categories.txt")
        self.category2label = dict()
        self.label2category = dict()
        with open(self.categories_path, "r") as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                category = line.replace("\n", "")
                if category not in ignore_classes:
                    self.category2label[category] = i
                    i += 1
                    self.label2category[i] = category
        self.num_classes = i
        val_file = os.path.join(self.label_directory, "validate" + str(dataset_id) + ".txt")
        test_file = os.path.join(self.label_directory, "test" + str(dataset_id) + ".txt")
        train_file = os.path.join(self.label_directory, "train" + str(dataset_id) + ".txt")

        self.validate = PartialMincDataset(val_file, self.category2label)
        self.test = PartialMincDataset(test_file, self.category2label)
        self.train = PartialMincDataset(train_file, self.category2label, augmentation=augmentation)


def create_loader(dataset, batch_size=32, num_workers=0):
    indices = list(range(len(dataset)))
    sampler = SubsetRandomSampler(indices)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       sampler=sampler, shuffle=False, num_workers=num_workers)


class MincLoader:
    def __init__(self, minc_dataset: MincDataset, batch_size=32):
        # TODO do data augmentation
        self.validate_loader = create_loader(minc_dataset.validate, batch_size=batch_size)
        self.train_loader = create_loader(minc_dataset.train, batch_size=batch_size)
        self.test_loader = create_loader(minc_dataset.train, batch_size=batch_size)


def save_invalid_minc2500_files(minc_dataset: MincDataset):
    minc_dataset.validate.set_get_function(debug=True)
    minc_dataset.train.set_get_function(debug=True)
    minc_dataset.test.set_get_function(debug=True)
    minc_loader = MincLoader(minc_dataset)

    for i in minc_loader.test_loader:
        pass
    minc_dataset.test.save_invalid_files()
    print("saved invalid test files")

    for i in minc_loader.validate_loader:
        pass
    minc_dataset.validate.save_invalid_files()
    print("saved invalid validation files")

    for i in minc_loader.train_loader:
        pass
    minc_dataset.train.save_invalid_files()
    print("saved invalid train files")


def replace_bw_image_with_rgb_equivalent(invalid_files):
    """

    Args:
        invalid_files: list of paths to black and white images that are to be converted to 3 channel images

    Returns:
        None. Overwrites bw images to same-appearance 3 channel images.
    """
    with open(invalid_files, "r") as f:
        for i in f:
            imfile = i.replace("\n", "")
            # new_im = np.zeros(imsize, dtype=np.uint8)
            new_im = np.zeros((362, 362, 3))
            with Image.open(imfile) as img:
                original_shape = np.array(img).shape
                imsize = (*original_shape, 3)
                print("original shape:", original_shape,"calculated target shape", imsize)
                assert len(original_shape) == 2, "input image isn't bw (2D)"
                new_im[:, :, 0] = img
                new_im[:, :, 1] = img
                new_im[:, :, 2] = img
                # print("new shape:", new_im.shape)
            newpilim = Image.fromarray(new_im)
            newpilim.save(imfile)
            print("saved image", imfile)


if __name__ == "__main__":
    # this_dir = os.path.dirname(__file__)
    # labels_dir = os.path.join(this_dir, "../../dataset_generation/MINC/minc-2500/labels")
    root_dir = "/home.nfs/hartvjir/MINC/minc-2500/"
    new_minc = MincDataset(root_dir, augmentation=True)
    minc_loader = MincLoader(new_minc)
    counter = 0
    for data, labels in minc_loader.train_loader:
        for im in data:
            im = Image.fromarray(np.transpose(np.array(im)*255.0, (1,2,0)).astype(np.uint8))
            im.save("imyo"+str(counter)+".jpg")
            # im.show()
            counter += 1
            if counter > 64:
                break
        if counter > 64:
            break
    # uncomment \/ in this order to find and convert 1 channel images to 3 channel images:
    # save_invalid_minc2500_files(new_minc)
    # replace_bw_image_with_rgb_equivalent("/home.nfs/hartvjir/invalid_files.txt")
