import torch

from dptdepth_model.configuration_dptdepth import DPTDepthConfig
from dptdepth_model.image_processor_dptdepth import DPTDepthImageProcessor
from dptdepth_model.modeling_dptdepth import DPTDepthModel

DPTDepthConfig.register_for_auto_class()
DPTDepthModel.register_for_auto_class("AutoModel")
DPTDepthImageProcessor.register_for_auto_class("AutoImageProcessor")

config = DPTDepthConfig()
model = DPTDepthModel(config)
model.model.load_state_dict(
    torch.load("./model_pths/omnidata_rgb2depth_dpt_hybrid.pth", map_location="cpu")
)
image_processor = DPTDepthImageProcessor()

model.push_to_hub("RGBD-SOD/dptdepth")
image_processor.push_to_hub("RGBD-SOD/dptdepth")
