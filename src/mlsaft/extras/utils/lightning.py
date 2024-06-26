import logging

import torch
from pytorch_lightning import LightningModule


def load_lightningmodule_from_checkpoint(ckpt_path: str, lit_model: LightningModule):
    """Load the state dict from a checkpoint into a LightningModule"""
    logger = logging.getLogger(__name__)
    loaded_state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
    loaded_model_state_dict = loaded_state_dict["state_dict"]
    model_state_dict = lit_model.state_dict()
    for param_name, param_value in loaded_model_state_dict.items():
        check_1 = param_name in model_state_dict
        check_2 = check_1 and model_state_dict[param_name].shape == param_value.shape
        if check_1 and check_2:
            model_state_dict[param_name] = param_value
        elif check_1 and not check_2:
            logger.info(
                "Skipping loading parameter %s because shapes do not match",
                param_name,
            )
    lit_model.load_state_dict(model_state_dict)
