import argparse
import os

# import yaml
from PIL import Image
import numpy as np
import torch

from .dataset_loader import MincDataset, MincLoader, get_free_gpu
# from .material_utils import material_str2raw_id, material_raw_id2str, material_ipalm_ignore, material_all_str
from .mapping_utils import *
# from .patch_architecture import MaterialPatchNet
from .net import MobileNetV3Large
import typing
from typing import Union, List, Tuple
#import pytorch_lightning

# from urllib import request
# import tarfile

__all__= ["get_classes_above_threshold", "map_from_selection", "to_numpy_cpu", "create_id2str", "get_top_indices", "get_labels_from_indices", "get_top_labels", "get_probabilities_from_selection"]

def main(args):
    root_dir = os.path.join("/home.nfs/hartvjir/MINC/minc-2500/")

    minc_dataset = MincDataset(root_dir)
    minc_loader = MincLoader(minc_dataset, batch_size=1)
    # ignore non-ipalm materials:
    ipalm_ids = [i for i in range(len(material_all_str)) if material_all_str[i] not in material_ipalm_ignore]
    # print(ipalm_ids)
    # exit()
    model_path = "saved_model.pth"
    if args.model is not None:
        model_path = args.model[0]
        print(args.model)
    else:
        print(args.model)
    model = MobileNetV3Large(n_classes=minc_dataset.num_classes)  # , input_size=362)
    model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available() and not args.usecpu:
        model = model.cuda()
    else:
        print("using cpu!")
    # for i in model.modules():
    #     print(i)
    model.eval()     # Optional when not using Model Specific layer
    if args.imagesfolder:
        images = os.listdir(args.imagesfolder)
        images = [i for i in images if ".jpg" in i or ".png" in i]
        print(os.listdir(args.imagesfolder))
        print(ipalm_ids)
        with open(os.path.join(args.imagesfolder, "results.txt"), "w") as results:
            # results.write(image+":\n"+str(l)+"\n")
            for image in images:
                with Image.open(os.path.join(args.imagesfolder, image)) as f:
                    f = f.crop(get_square_crop(f.size))
                    f = f.resize((362, 362))
                    data = np.transpose(np.array(f).astype('f4'), (2, 0, 1)) / 255.0
                    data = torch.from_numpy(data)
                    data.unsqueeze_(0)
                    # print(data.size())
                    if torch.cuda.is_available() and not args.usecpu:
                        data = data.cuda()
                    # Forward Pass
                    target = model(data)
                    probs = get_probabilities_from_selection(target, ipalm_ids)
                    print(image, ": ", end="")
                    class_probs = list(i for i in zip(map_from_selection(ipalm_ids, material_raw_id2str), to_numpy_cpu(probs)))
                    # print(class_probs)
                    # * means unzip, aka remove the parentheses as if in code
                    print(*get_classes_above_threshold(class_probs))
                    # indices = get_top_indices(target)
                    # l = get_labels_from_indices(indices, id2category)
                    # results.write(image+":\n"+str(l)+"\n")
            print()
    else:
        test_minc_dataset(model, minc_loader)


def get_classes_above_threshold(class_prob_tpl, threshold=0.1):  # Union[List[Tuple], Tuple[Tuple]], threshold=0.1):
    ret = list()
    for cls, prob in class_prob_tpl:
        if prob >= threshold:
            ret.append((cls, prob))
    return ret


def map_from_selection(selected_ids, id_mapping):
    # e.g. selected_ids: [2, 3, 5, 6] => only want items from these indices
    ret = list()
    for i in range(len(selected_ids)):
        ret.append(id_mapping[selected_ids[i]])
    return ret


def to_numpy_cpu(x):
    return x.cpu().detach().numpy()


def get_probabilities_from_selection(y, selection):
    ret = y[0]
    m = torch.nn.Softmax(dim=0)
    ret = m(ret[selection])  # select only the wanted classes and THEN perform softmax on them
    return ret


def get_square_crop(im_size):
    width, height = im_size  # as opposed to numpy, which gives height, width
    new_width = min(im_size)
    new_height = new_width
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    return left, top, right, bottom


def test_minc_dataset(model, minc_loader: MincLoader):
    for data, labels in minc_loader.test_loader:
        im = (data.numpy()[0]*255.0).astype('uint8')
        im = np.transpose(im, (1, 2, 0))
        # Transfer Data to GPU if available
        if torch.cuda.is_available() and not args.usecpu:
            data, labels = data.cuda(), labels.cuda()

        # Forward Pass
        target = model(data)
        # labels = get_top_labels(target, id2str)
        indices = get_top_indices(target)
        l = get_labels_from_indices(indices, material_raw_id2str)

        print(l)
        im = Image.fromarray(im)
        # im_name = ""
        # im.save("yoyoy.jpg")


def create_id2str(str2id):
    ret = dict()
    for k, v in enumerate(str2id):
        ret[k] = v
    return ret


def get_top_indices(y, number=3):
    """

    Args:
        y: output of the network
        number: number of indices with the highest values to be returned

    Returns: list of indices in descending order
    """
    return np.flip(np.argsort(y.cpu().detach().numpy()[0]))[:number]


def get_labels_from_indices(indices, id2str):
    ret = list()
    for i in range(len(indices)):
        ret.append(id2str[indices[i]])
    return ret


def get_top_labels(y, id2str, number=3):
    indices = np.flip(np.argsort(y.cpu().detach().numpy()[0]))
    ret = list()
    for i in range(number):
        ret.append(id2str[indices[i]])
    return ret


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", nargs="?", help="")  # 0 or 1 args
    parser.add_argument("--usecpu", type=bool, nargs="?", help="")  # 0 or 1 args
    parser.add_argument("--imagesfolder", type=str, nargs="?", help="")  # 0 or 1 args
    args = parser.parse_args()
    main(args)


