from typing import Dict, Optional, Tuple

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL.Image import Image
from torch import Tensor
from transformers.image_processing_utils import BaseImageProcessor

INPUT_IMAGE_SIZE = (352, 352)

transform = transforms.Compose(
    [
        transforms.Resize(
            INPUT_IMAGE_SIZE,
            interpolation=TF.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ]
)


class DPTDepthImageProcessor(BaseImageProcessor):
    model_input_names = ["dptdepth_preprocessor"]

    def __init__(self, testsize: Optional[int] = 352, **kwargs) -> None:
        super().__init__(**kwargs)
        self.testsize = testsize

    def preprocess(
        self, inputs: Dict[str, Image], **kwargs  # {'rgb': ... }
    ) -> Dict[str, Tensor]:
        rgb: Tensor = transform(inputs["rgb"])
        return dict(rgb=rgb.unsqueeze(0))

    def postprocess(
        self, logits: Tensor, size: Tuple[int, int], **kwargs
    ) -> np.ndarray:
        logits: Tensor = F.upsample(
            logits, size=size, mode="bilinear", align_corners=False
        )
        res: np.ndarray = logits.squeeze().data.cpu().numpy()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        return res
