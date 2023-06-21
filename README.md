# <p align=center>`DPT-Depth`</p> 

Example how to run:

```python
from transformers import AutoImageProcessor, AutoModel
from typing import Dict, Tuple

import numpy as np
from matplotlib import cm
from PIL import Image
from torch import Tensor

model = AutoModel.from_pretrained(
    "RGBD-SOD/dptdepth", trust_remote_code=True, cache_dir="model_cache"
)
image_processor = AutoImageProcessor.from_pretrained(
    "RGBD-SOD/dptdepth", trust_remote_code=True, cache_dir="image_processor_cache"
)


def inference(rgb: Image.Image) -> Tuple[Image.Image, Image.Image]:
    rgb = rgb.convert(mode="RGB")

    preprocessed_sample: Dict[str, Tensor] = image_processor.preprocess(
        {
            "rgb": rgb,
        }
    )

    output: Dict[str, Tensor] = model(preprocessed_sample["rgb"])
    postprocessed_sample: np.ndarray = image_processor.postprocess(
        output["logits"], [rgb.size[1], rgb.size[0]]
    )

    raw_pred = Image.fromarray((postprocessed_sample * 256).astype(np.uint8))
    print(np.min(raw_pred), np.max(raw_pred))
    visualized_pred = Image.fromarray(
        np.uint8(cm.gist_earth(postprocessed_sample) * 255)
    )
    return raw_pred, visualized_pred


if __name__ == "__main__":
    origin = Image.open("origin.png")
    raw_pred, visualized_pred = inference(origin)
    raw_pred.save("raw_pred.png")
    visualized_pred.save("visualized_pred.png")
```
