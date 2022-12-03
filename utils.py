import os

import numpy as np
from wandb.integration.sb3 import WandbCallback


def ensure_dir(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


class WandbCallbackWithFood(WandbCallback):
    def _on_rollout_end(self):
        if len(self.model.ep_info_buffer) == 0:
            self.model.logger.record("rollout/ep_food_taken_mean", 0)
        else:
            self.model.logger.record(
                "rollout/ep_food_taken_mean", np.mean([info['food_taken'] for info in self.model.ep_info_buffer]))
