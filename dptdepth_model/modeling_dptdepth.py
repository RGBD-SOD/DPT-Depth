from typing import Dict, Optional

from torch import Tensor, nn
from transformers import PreTrainedModel

from .configuration_dptdepth import DPTDepthConfig
from .models import DPTDepthModel as DPTDepth


class DPTDepthModel(PreTrainedModel):
    """
    The line that sets the config_class is not mandatory,
    unless you want to register your model with the auto classes
    """

    config_class = DPTDepthConfig

    def __init__(self, config: DPTDepthConfig):
        super().__init__(config)
        self.model = DPTDepth()
        self.loss = nn.L1Loss()

    """
    You can have your model return anything you want, 
    but returning a dictionary with the loss included when labels are passed, 
    will make your model directly usable inside the Trainer class. 
    Using another output format is fine as long as you are planning on 
    using your own training loop or another library for training.
    """

    def forward(self, rgbs: Tensor, gts: Optional[Tensor] = None) -> Dict[str, Tensor]:
        logits = self.model(rgbs)
        if gts is not None:
            loss = self.loss(logits, gts)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
