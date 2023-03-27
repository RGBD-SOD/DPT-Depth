from typing import Dict

import numpy as np
from datasets import load_dataset
from matplotlib import cm
from PIL import Image
from torch import Tensor
from transformers import AutoImageProcessor, AutoModel

model = AutoModel.from_pretrained("RGBD-SOD/dptdepth", trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained(
    "RGBD-SOD/dptdepth", trust_remote_code=True
)
dataset = load_dataset("RGBD-SOD/test", "v1", split="train", cache_dir="data")

index = 0

"""
Get a specific sample from the dataset

sample = {
    'depth': <PIL.PngImagePlugin.PngImageFile image mode=L size=640x360>, 
    'rgb': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=640x360>, 
    'gt': <PIL.PngImagePlugin.PngImageFile image mode=L size=640x360>, 
    'name': 'COME_Train_5'
}
"""
sample = dataset[index]

depth: Image.Image = sample["depth"]
rgb: Image.Image = sample["rgb"]
gt: Image.Image = sample["gt"]
name: str = sample["name"]


"""
1. Preprocessing step

preprocessed_sample = {
    'rgb': tensor([[[[-0.8507, ....0365]]]]), 
}
"""
preprocessed_sample: Dict[str, Tensor] = image_processor.preprocess(sample)

"""
2. Prediction step

output = {
    'logits': tensor([[[[-5.1966, ...ackward0>)
}
"""
output: Dict[str, Tensor] = model(preprocessed_sample["rgb"])

"""
3. Postprocessing step
"""
postprocessed_sample: np.ndarray = image_processor.postprocess(
    output["logits"], [sample["gt"].size[1], sample["gt"].size[0]]
)
prediction = Image.fromarray(np.uint8(cm.gist_earth(postprocessed_sample) * 255))

"""
Show the predicted salient map and the corresponding ground-truth(GT)
"""
prediction.show()
gt.show()
