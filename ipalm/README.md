# Utilities for extra material & category classification for the IPALM project

This folder contains all the custom files added as part of the ipalm project.

### File summaries
**Note**: This list contains file only necessary for runtime
- **folder** `models/` - this should contain the trained detectron and mobilenet models. They are saved [here](https://drive.google.com/drive/folders/1iu_YYbbb6iiyAiCXsjhjMEMQx-KsLr7y?usp=sharing)

Necessary files on disk:
```
model_path="ipalm/models/model_final.pth",
material_model_path="ipalm/models/saved_model.pth",
confusion_matrices="ipalm/config/confusion_matrices.json"
```
- `andrej_output_format.txt` - contains a sample file in the format that andrej uses in his bayesian network
- `detectron2_to_mobilenet.py`
  - `get_materials_from_patches` - takes in a list of images (bboxes) and a list of probabilities for each image. The probabilities are only counted when they are larger than 0.1. (0.1, 0.0, 0.5, 0.0, ...); sum(probs) = 1.
- `intermediate_data.py`:
<img src="https://i.imgur.com/pqltyJA.png" width=500>

An image is plugged into Detectron2, we get instance segmentation, bounding boxes and confidence score for the highest category. We cut out the bounding boxes from the original image and input each cutout back into the original Detectron2 - but with a lower threshold score to get many outputs - and into MobileNet to get material classification.


- `andrej_logic.py` - main class interface of this script. Contains [CatMatPredictor](https://github.com/Hartvi/Detectron2-mobilenet/blob/4f7e5f1a54f5be6b773dddd3905443f9d35c0d74/andrej_logic.py#L88) which has the method get_andrej
- `mapping_utils.py` - contains all the mappings between the 66 old VIR detectron2 outputs to just the classes (36), chosen materials from MINC (8). The 66 old VIR outputs are tuples: (category - material) therefore they can be mapped to 36 categories and roughly 8 materials from MINC.
- `test.py` - contains some testing/evaluation functions
- `setup_utils.py` - contains the setup function for detectron2

