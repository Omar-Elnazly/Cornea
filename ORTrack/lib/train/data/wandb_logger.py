from collections import OrderedDict

# ===============================
# SAFE WANDB DISABLED VERSION
# ===============================

class WandbWriter:
    def __init__(self, *args, **kwargs):
        print("✅ WandB disabled — training continues normally.")

    def write_log(self, stats: OrderedDict, epoch=-1):
        pass