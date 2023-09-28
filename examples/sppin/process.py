#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 09:28:09 2023

@author: myrthebuser
SPPIN Example Submission

Please only change things on the places were it is indicated.
"""

# Copyright 2023 Radboud University Medical Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from dataclasses import dataclass, make_dataclass
from enum import Enum
from typing import Any, Dict
from pathlib import Path

import SimpleITK
import ants
from unet3d.scripts.predict import run_inference
import nibabel as nib
import numpy as np

INPUT_PREFIX = Path('/input')
OUTPUT_PREFIX = Path('/output')

class IOKind(str, Enum):
    JSON = "JSON"
    IMAGE = "Image"
    FILE = "File"

class InterfaceKind(str, Enum):
    def __new__(cls, value, annotation, io_kind):
        member = str.__new__(cls, value)
        member._value_ = value
        member.annotation = annotation
        member.io_kind = io_kind
        return member

    STRING = "String", str, IOKind.JSON
    INTEGER = "Integer", int, IOKind.JSON
    FLOAT = "Float", float, IOKind.JSON
    BOOL = "Bool", bool, IOKind.JSON
    ANY = "Anything", Any, IOKind.JSON
    CHART = "Chart", Dict[str, Any], IOKind.JSON

    # Annotation Types
    TWO_D_BOUNDING_BOX = "2D bounding box", Dict[str, Any], IOKind.JSON
    MULTIPLE_TWO_D_BOUNDING_BOXES = "Multiple 2D bounding boxes", Dict[str, Any], IOKind.JSON
    DISTANCE_MEASUREMENT = "Distance measurement", Dict[str, Any], IOKind.JSON
    MULTIPLE_DISTANCE_MEASUREMENTS = "Multiple distance measurements", Dict[str, Any], IOKind.JSON
    POINT = "Point", Dict[str, Any], IOKind.JSON
    MULTIPLE_POINTS = "Multiple points", Dict[str, Any], IOKind.JSON
    POLYGON = "Polygon", Dict[str, Any], IOKind.JSON
    MULTIPLE_POLYGONS = "Multiple polygons", Dict[str, Any], IOKind.JSON
    LINE = "Line", Dict[str, Any], IOKind.JSON
    MULTIPLE_LINES = "Multiple lines", Dict[str, Any], IOKind.JSON
    ANGLE = "Angle", Dict[str, Any], IOKind.JSON
    MULTIPLE_ANGLES = "Multiple angles", Dict[str, Any], IOKind.JSON
    ELLIPSE = "Ellipse", Dict[str, Any], IOKind.JSON
    MULTIPLE_ELLIPSES = "Multiple ellipses", Dict[str, Any], IOKind.JSON

    # Choice Types
    CHOICE = "Choice", int, IOKind.JSON
    MULTIPLE_CHOICE = "Multiple choice", int, IOKind.JSON

    # Image types
    IMAGE = "Image", bytes, IOKind.IMAGE
    SEGMENTATION = "Segmentation", bytes, IOKind.IMAGE
    HEAT_MAP = "Heat Map", bytes, IOKind.IMAGE

    # File types
    PDF = "PDF file", bytes, IOKind.FILE
    SQREG = "SQREG file", bytes, IOKind.FILE
    THUMBNAIL_JPG = "Thumbnail jpg", bytes, IOKind.FILE
    THUMBNAIL_PNG = "Thumbnail png", bytes, IOKind.FILE
    OBJ = "OBJ file", bytes, IOKind.FILE
    MP4 = "MP4 file", bytes, IOKind.FILE

    # Legacy support
    CSV = "CSV file", str, IOKind.FILE
    ZIP = "ZIP file", bytes, IOKind.FILE

@dataclass
class Interface:
    slug: str
    relative_path: str
    kind: InterfaceKind

    @property
    def kwarg(self):
        return self.slug.replace("-", "_").lower()

    def load(self):
        if self.kind.io_kind == IOKind.JSON:
            return self._load_json()
        elif self.kind.io_kind == IOKind.IMAGE:
            return self._load_image()
        elif self.kind.io_kind == IOKind.FILE:
            return self._load_file()
        else:
            raise AttributeError(f"Unknown io kind {self.kind.io_kind!r} for {self.kind!r}")

    def save(self, *, data):
        if self.kind.io_kind == IOKind.JSON:
            return self._save_json(data=data)
        elif self.kind.io_kind == IOKind.IMAGE:
            return self._save_image(data=data)
        elif self.kind.io_kind == IOKind.FILE:
            return self._save_file(data=data)
        else:
            raise AttributeError(f"Unknown io kind {self.kind.io_kind!r} for {self.kind!r}")

    def _load_json(self):
        with open(INPUT_PREFIX / self.relative_path, "r") as f:
            return json.loads(f.read())

    def _save_json(self, *, data):
        with open(OUTPUT_PREFIX / self.relative_path, "w") as f:
            f.write(json.dumps(data))

    def _load_image(self):
        input_directory = INPUT_PREFIX / self.relative_path

        mha_files = {f for f in input_directory.glob("*.mha") if f.is_file()}

        if len(mha_files) == 1:
            mha_file = mha_files.pop()
            return SimpleITK.ReadImage(mha_file)
        elif len(mha_files) > 1:
            raise RuntimeError(
                f"More than one mha file was found in {input_directory!r}"
            )
        else:
            print(input_directory)
            raise NotImplementedError

    def _save_image(self, *, data):
        output_directory = OUTPUT_PREFIX / self.relative_path

        output_directory.mkdir(exist_ok=True, parents=True)

        SimpleITK.WriteImage(data, output_directory / "overlay.mha")

    @property
    def _file_mode_suffix(self):
        if self.kind.annotation == str:
            return ""
        elif self.kind.annotation == bytes:
            return "b"
        else:
            raise AttributeError(f"Unknown annotation {self.kind.annotation!r} for {self.kind!r}")

    def _load_file(self):
        with open(INPUT_PREFIX / self.relative_path, f"r{self._file_mode_suffix}") as f:
            return f.read()

    def _save_file(self, *, data):
        with open(OUTPUT_PREFIX / self.relative_path, f"w{self._file_mode_suffix}") as f:
            f.write(data)

INPUT_INTERFACES = [
    Interface(slug="pediatric-abdominal-mri-t1", relative_path="images/abdominal-mr_t1", kind=InterfaceKind.IMAGE),
    Interface(slug="pediatric-abdominal-mri-t2", relative_path="images/abdominal-mr_t2", kind=InterfaceKind.IMAGE),
    Interface(slug="pediatric-abdominal-mri-dwi-b0", relative_path="images/abdominal-mr_b0", kind=InterfaceKind.IMAGE),
    Interface(slug="pediatric-abdominal-mri-dwi-b100", relative_path="images/abdominal-mr_b100", kind=InterfaceKind.IMAGE),
]

OUTPUT_INTERFACES = [
    Interface(slug="mri-segmentation-of-pediatric-neuroblastoma", relative_path="images/neuroblastoma-segmentation", kind=InterfaceKind.SEGMENTATION),
]

Inputs = make_dataclass(cls_name="Inputs", fields=[(inpt.kwarg, inpt.kind.annotation) for inpt in INPUT_INTERFACES])

Outputs = make_dataclass(cls_name="Outputs", fields=[(output.kwarg, output.kind.annotation) for output in OUTPUT_INTERFACES])

def load() -> Inputs:
    return Inputs(
        **{interface.kwarg: interface.load() for interface in INPUT_INTERFACES}
    )

def predict(*, inputs: Inputs) -> Outputs:
    """ This is the place where you can implement your own algoritme. Please note taht the outputs have to match the size of the T1 images 
    to properly be evaluated. """
    
    # Define the inputs 
    t1 = inputs.pediatric_abdominal_mri_t1
    t2 = inputs.pediatric_abdominal_mri_t2
    dwi_b0 = inputs.pediatric_abdominal_mri_dwi_b0
    dwi_b100 = inputs.pediatric_abdominal_mri_dwi_b100

    #  0. write the files so that I make sure they are read correctly
    _tmp_dir = "/tmp/3DUNetCNN"
    os.makedirs(_tmp_dir, exist_ok=True)
    image_filenames = list()
    for image, name in zip((t1, t2, dwi_b0, dwi_b100), ("t1", "t2", "b0", "b100")):
        out_filename = os.path.join(_tmp_dir, f"{name}.nii.gz")
        SimpleITK.WriteImage(image, out_filename)
        image_filenames.append(out_filename)

    del t1, t2, dwi_b0, dwi_b100

    #  1. register the images to the T1w image
    fixed = ants.image_read(image_filenames[0])
    final_image_filenames = [image_filenames[0]]
    for moving_filename in image_filenames[1:]:
        moving = ants.image_read(moving_filename)
        result = ants.registration(fixed=fixed, moving=moving, type_of_transform="QuickRigid")
        out_filename = os.path.join(_tmp_dir, f"aligned_{os.path.basename(moving_filename)}")
        ants.image_write(result["warpedmovout"], out_filename)
        final_image_filenames.append(out_filename)

    del result

    #  2. predict the segmentation image
    with open("/model/sppin_config.json", "r") as op:
        config = json.load(op)
    # make sure the model doesn't predict any other filenames
    keys_to_remove = list()
    for key in config:
        if "_filenames" in key:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        config.pop(key)
    config["test_filenames"] = [{"image": final_image_filenames}]

    prediction_images = list()
    model_filenames = ["/model/model1.pth", "/model/model2.pth", "/model/model3.pth", "/model/model4.pth",
                       "/model/model5.pth"]
    for model_filename in model_filenames:
        model_basename = os.path.basename(model_filename).split(".")[0]
    
        _output_dir = os.path.join(_tmp_dir, "predictions", model_basename)
        os.makedirs(_output_dir, exist_ok=True)
    
        output_filenames = run_inference(config=config,
                                         output_directory=_output_dir,
                                         model_filename=model_filename,
                                         group="test", activation="sigmoid",
                                         system_config={"n_gpus": 1, "n_workers": 0})
        prediction_images.append(nib.load(output_filenames[0]))

    prediction_filename = os.path.join(_tmp_dir, "predictions", "prediction.nii.gz")
    os.makedirs(os.path.dirname(prediction_filename), exist_ok=True)
    combined_data = np.stack([np.asarray(im.dataobj) for im in prediction_images])
    mean_data = np.mean(combined_data, axis=0)
    prediction_image = prediction_images[0].__class__(dataobj=mean_data, affine=prediction_images[0].affine)
    prediction_image.to_filename(prediction_filename)

    #  3. read in the output images back into sitk
    prediction = SimpleITK.ReadImage(prediction_filename)

    # Threshold and select the largest connected component
    binary_image = SimpleITK.BinaryThreshold(
            image1=prediction, lowerThreshold=0.5, insideValue=1, outsideValue=0
        )
    component_image = SimpleITK.ConnectedComponent(binary_image)
    sorted_component_image = SimpleITK.RelabelComponent(component_image, sortByObjectSize=True)
    segmentation = sorted_component_image == 1
    
    outputs = Outputs(
        mri_segmentation_of_pediatric_neuroblastoma = segmentation
    )

    return outputs

def save(*, outputs: Outputs) -> None:
    for interface in OUTPUT_INTERFACES:
        interface.save(data=getattr(outputs, interface.kwarg))

def main() -> int:
    inputs = load()
    outputs = predict(inputs=inputs)
    save(outputs=outputs)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
