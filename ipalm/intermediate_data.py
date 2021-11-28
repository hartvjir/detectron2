import collections
from typing import Tuple, Union, List
import numpy as np
from .mapping_utils import *


class SimpleInstance:
    """
    Detected instance inside an image
    > processes like so: output.pred_classes, output.scores, output.pred_boxes.tensor
    """
    def __init__(self, raw_id, score, bbox):
        self.category = raw_id_to_shortened_id[raw_id]
        self.bbox = bbox
        self.score = score

    def __str__(self):
        ret = ""
        ret += "\n" +str(shortened_id_to_str[self.category])
        ret += "\n "+str(self.score)
        ret += "\n "+str(self.bbox)
        return ret


class IntermediateOutput:
    """
    Contains all instances for one image
    """
    def __init__(self, simple_output: List[SimpleInstance]):
        self.simple_output = simple_output

    def __getitem__(self, item):
        return self.simple_output[item]

    def __iter__(self):
        yield from self.simple_output

    def __len__(self):
        return len(self.simple_output)

    def __str__(self):
        ret = ""
        for out in self:
            ret += "\n"+str(out)
        return ret


class IntermediateOutputs:
    """
    Contains intermediate outputs for a list of images
    """
    def __init__(self, intermediate_output: List[IntermediateOutput]):
        self.intermediate_output = intermediate_output

    def __getitem__(self, item):
        return self.intermediate_output[item]

    def __iter__(self):
        yield from self.intermediate_output

    def __len__(self):
        return len(self.intermediate_output)

    def __str__(self):
        ret = ""
        for out in self:
            ret += "\n" + str(out)
        return ret


class IntermediateInput:
    """
    Contains intermediate inputs for one image
    """
    def __init__(self, intermediate_input: Union[List, np.ndarray]):
        self.intermediate_input = intermediate_input

    def __getitem__(self, item):
        return self.intermediate_input[item]

    def __iter__(self):
        yield from self.intermediate_input

    def __len__(self):
        return len(self.intermediate_input)

    def __str__(self):
        ret = ""
        for inp in self:
            ret += "\n" + str(inp)


class IntermediateInputs:
    """
    Contains a list of intermediate inputs for a list of images
    """
    def __init__(self, mobile_inputs: List[IntermediateInput], detectron_inputs: List[IntermediateInput]):
        self.mobile_inputs = mobile_inputs
        self.detectron_inputs = detectron_inputs
        self.input_zip = tuple(zip(self.mobile_inputs, self.detectron_inputs))

    def __getitem__(self, item):
        return self.input_zip[item]

    def __iter__(self):
        yield from self.input_zip

    def __len__(self):
        return len(self.mobile_inputs)

    def __str__(self):
        ret = ""
        for inp in self:
            ret += "\n" + str(inp)
        return ret


class IntermediateData:
    """
    Contains a list of intermediate outputs for a list of images.
    And: a list of intermediate inputs for the same list of images.
    """
    def __init__(self, im_names: List[str], outputs: IntermediateOutputs, inputs: IntermediateInputs):
        self.im_names = im_names
        self.outputs = outputs
        self.inputs = inputs

    def __str__(self):
        ret = ""
        for name, outs, inps in zip(self.im_names, self.outputs, self.inputs):
            ret += "\n" + str(name) + ":\n " + str(inps) + "\n " + str(outs)
        return ret


class TypedList(collections.MutableSequence):

    def __init__(self, oktypes, *args):
        self.oktypes = oktypes
        self.list = list()
        self.extend(list(args))

    def check(self, v):
        if not isinstance(v, self.oktypes):
            raise TypeError(v)

    def __len__(self): return len(self.list)

    def __getitem__(self, i): return self.list[i]

    def __delitem__(self, i): del self.list[i]

    def __setitem__(self, i, v):
        self.check(v)
        self.list[i] = v

    def insert(self, i, v):
        self.check(v)
        self.list.insert(i, v)

    def __str__(self):
        return str(self.list)


