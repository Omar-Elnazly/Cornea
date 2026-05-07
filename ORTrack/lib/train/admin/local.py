import os

class EnvironmentSettings:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))

        self.workspace_dir = os.getenv("ORTRACK_PATH", project_root)

        self.tensorboard_dir = os.path.join(self.workspace_dir, "tensorboard")
        self.pretrained_networks = os.path.join(self.workspace_dir, "pretrained_networks")

        self.lasot_dir = os.path.join(self.workspace_dir, "data", "lasot")
        self.got10k_dir = os.path.join(self.workspace_dir, "data", "got10k", "train")
        self.got10k_val_dir = os.path.join(self.workspace_dir, "data", "got10k", "val")

        self.lasot_lmdb_dir = os.path.join(self.workspace_dir, "data", "lasot_lmdb")
        self.got10k_lmdb_dir = os.path.join(self.workspace_dir, "data", "got10k_lmdb")

        self.trackingnet_dir = os.path.join(self.workspace_dir, "data", "trackingnet")
        self.trackingnet_lmdb_dir = os.path.join(self.workspace_dir, "data", "trackingnet_lmdb")

        self.coco_dir = os.path.join(self.workspace_dir, "data", "coco")
        self.coco_lmdb_dir = os.path.join(self.workspace_dir, "data", "coco_lmdb")

        self.imagenet_dir = os.path.join(self.workspace_dir, "data", "vid")
        self.imagenet_lmdb_dir = os.path.join(self.workspace_dir, "data", "vid_lmdb")


        self.lvis_dir = ""
        self.sbd_dir = ""
        self.imagenetdet_dir = ""
        self.ecssd_dir = ""
        self.hkuis_dir = ""
        self.msra10k_dir = ""
        self.davis_dir = ""
        self.youtubevos_dir = ""

    
        self.external_uav_dir = os.path.join(self.workspace_dir, "external_uav")
        self.metadata_dir = os.path.join(self.workspace_dir, "metadata")
        self.data_dir = os.path.join(self.workspace_dir, "data_small")