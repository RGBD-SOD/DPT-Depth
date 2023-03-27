from typing import List

from transformers import PretrainedConfig

"""
The configuration of a model is an object that 
will contain all the necessary information to build the model.
The three important things to remember when writing you own configuration are the following:
- you have to inherit from PretrainedConfig,
- the __init__ of your PretrainedConfig must accept any kwargs,
- those kwargs need to be passed to the superclass __init__.
"""


class DPTDepthConfig(PretrainedConfig):

    """
    Defining a model_type for your configuration is not mandatory,
    unless you want to register your model with the auto classes."""

    model_type = "dptdepth"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
