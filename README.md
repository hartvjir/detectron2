# Object detection, category classification and material classification for IPALM

Part of the IPALM project, this is a fusion of a MobilenetV3 trained on the smaller MINC2500 dataset and the default Detectron2 InstanceSegmentor trained on COCO, [ShopVRB](https://michaal94.github.io/SHOP-VRB/), [YCB](https://www.ycbbenchmarks.com/) and a few custom images.

- Repository for training MobileNet on MINC2500 [here](https://gitlab.fel.cvut.cz/body-schema/ipalm/ipalm-vir2020-object-category-from-image/-/tree/master/code/patch_based_material_recognition) (gitlab).

- Repository for creating test data and evaluating test data for this project to create the confusion matrix from which the precision is calculated [here](https://github.com/Hartvi/ImPointAnnotator) (github).

- Files added for the ipalm project, such as the material classification script are located in [ipalm/](https://github.com/Hartvi/Detectron2-mobilenet/tree/main/ipalm)

# TODO
- remove personal references => remove Andrej ;-;
- add examples of how an input image is processed
- clarify the diagrams
- formalize the text
- list the materials that are used from MINC
- make a backup of this epo and make a fork of it from the original detectron repository
- etc

### Prerequisites
- Tested on Ubuntu 18.04 & Debian server
- Requires Linux with CUDA
- Versions of packages used:
  - `torchvision/0.9.1-fosscuda-2019b-PyTorch-1.8.0`
  - `OpenCV/3.4.8-fosscuda-2019b-Python-3.7.4`
  - `scikit-learn/0.21.3-fosscuda-2019b-Python-3.7.4`
  - `scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4`
  - and dependencies

### How to install:
1. Go to some folder A and: `git clone https://github.com/Hartvi/Detectron2-mobilenet`
    - This will create the folder called `Detectron2-mobilenet
2. Rename `Detectron2-mobilenet` to `detectron2`: `mv Detectron2-mobilenet detectron2`
3. In folder A: `python -m pip install -e detectron2 --user`
- All in one: `git clone https://github.com/Hartvi/Detectron2-mobilenet && mv Detectron2-mobilenet detectron2 && python -m pip install -e detectron2 --user`


### Short demo:

```
from detectron2 import andrej_logic

megapredictor = CatmatPredictor(threshold=0.6)
# folder with images: "images_input/[some_images]"
input_imgs = ["images_input/" + f for f in listdir("images_input") if isfile(join("images_input", f))]
# CatMatPredictor.get_andrej(raw_image[arr]/image_path[str]) returns a list of dictionaries for 
for inp_img in input_imgs:
    # this is a list of dicts in andrej format, see ipalm/andrej_output_format
    # optional argument: output_target="your_file_name.json" to save the dicts in json format
    predictions = megapredictor.get_andrej(inp_img)  
    # plot:
    quick_plot_bboxes(predictions, inp_img)
```


### Output interface
The outputs are going to be further processed downstream and they require precision and probabilities.

The COCO style dataset compiled during the VIR project has labels in the format `integer: "category - material"` in human readable terms. However, the MINC materials do not exactly correspond to the materials from the COCO-style datasets, so we selected 8 materials from MINC to use for this project. 

The requiredprecision is then calculated from the confusion matrix gained from running the networks on the test dataset that was used to train the basic Detectron2 for VIR. The first row and column of the matrix are ignored because those are the cases when the bounding boxes didn't contain any object in the image.

## Information flow
For a summary of the contents of the files added for the ipalm project, see the [ipalm/README.md](https://github.com/Hartvi/Detectron2-mobilenet/tree/main/ipalm#readme)

The high-level structure of the project is the following. The input image is fed into Detectron2 which is first used to locate objects of interest and its output data is saved. The bounding boxes gained from the first pass are extracted and plugged into Detectron2 (again) and also into MobileNet.
<div align=center>
    <img src="https://i.imgur.com/JcbV39e.png" alt="drawing" width="500"/><br>
    Figure 1. Information flow in project structure.
</div>
<br>

The following picture contains an explanation how categories are weighted. There are in total 2 passes of each bounding box through detectron. Therefore there is 1 bounding box that is then plugged back into detectron to get some more bounding boxes. The weight of the class initially detected by detectron is then `area_of_first_bbox*first_probability` + `area_of_nested_bbox*second_probability`. The weights of other classes are simply just `area_of_nested_bbox_of_class_i*second_probability_of_class_i`.


<div align=center>
    <img src="https://i.imgur.com/IpxOxNd.png" alt="drawing" width="500"/><br>
    Figure 2. Category and material probability calculation.
</div>

### Making of this project
The base of the project is the detectron2 framework's instance segmentation backbone by [facebookresearch](https://github.com/facebookresearch/detectron2). 
Initially we tried to add a material segmentation/classification head to the instance segmentation network, however that proved to be exceedingly confusing because of the structure of the project. See the [gitlab progress log](https://gitlab.fel.cvut.cz/body-schema/ipalm/ipalm-vir2020-object-category-from-image/-/blob/master/code/PROGRESS.md) for details on how it was unfeasible (for me) given the time constraints. Basically the detectron2's structure is such that I couldn't even discover any modifications of it **not made by facebook employees**.

We ended up using just the default instance segmentor from VIR, retrained because something in either a newer detectron2 or pytorch version changed something in the background. The bounding boxes gathered from detectron2 are then plugged back into detectron2 this time used as a category classifier and into a MobileNetV3 material classifier trained on the [MINC 2500](http://opensurfaces.cs.cornell.edu/publications/minc/) dataset.

### Encountered challenges
#### 1. Detectron2 project structure
I spent about 3 work weeks trying to add a ROIHead to detectron2 that would also classify the material properties of the object inside the bounding box, however adding a head to detectron required a non-trivial modification of the project. None of the IDEs let alone searching through the raw text helped very much in determining where the program was being executed. A rough ROIHead definition in a yaml file, calling the python class using its string name in the yaml, some configuration in the python script, some configuration is completely hidden, actual ROIHead code as a python class, backprop functions, dataset processing and formatting, nesting the classification into the bounding box, etc.

#### 2. Dataset formatting
The dataset formatting in COCO was done by Michal Pliska in the VIR subject. Formatting MINC2500 was only a minor issue of 2 of the 57500 images being single channel black and white, which crashed the training seemingly randomly.

## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing Detectron2

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
