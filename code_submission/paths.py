"""Central path configuration for every script in this package.

All scripts import their input locations from here, so there is ONE place that
defines where the data and weights live. Defaults assume the layout below; you
can override either root with an environment variable.

Expected layout (raw data is NOT shipped — place it here before running):

    code_submission/
      data/
        images/train/   *.jpg      competition training images
        images/val/     *.jpg      our local validation split (from prepare_yolo_obb.py)
        labels/train/   *.txt      YOLO-OBB labels: "0 x1 y1 ... x4 y4" (normalized)
        labels/val/     *.txt
        test_images/    *.jpg      the 226 official (unlabeled) competition test images
        shiprs_val_unseen/         (optional) ShipRSImageNet unseen set for §6.5/§7.4
          images/val/  labels/val/
      weights/
        best.pt                    from-scratch model
        best_finetuned.pt          fine-tuned model

Overrides:
    SHIP_DATA      -> data root      (default: <package>/data)
    SHIP_WEIGHTS   -> weights root   (default: <package>/weights)
"""
import os

PKG = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SHIP_DATA", os.path.join(PKG, "data"))
WEIGHTS_DIR = os.environ.get("SHIP_WEIGHTS", os.path.join(PKG, "weights"))

# raw competition download (original "cx cy w h theta" labels) — input to
# prepare_yolo_obb.py and quality_audit.ipynb
RAW_IMG_ROOT = os.path.join(DATA, "raw", "images")     # holds train/ (+ val/ if labeled)
RAW_LBL_ROOT = os.path.join(DATA, "raw", "labels")
RAW_IMG = os.path.join(RAW_IMG_ROOT, "train")
RAW_LBL = os.path.join(RAW_LBL_ROOT, "train")

IMG_TRAIN = os.path.join(DATA, "images", "train")
LBL_TRAIN = os.path.join(DATA, "labels", "train")
IMG_VAL = os.path.join(DATA, "images", "val")
LBL_VAL = os.path.join(DATA, "labels", "val")
TEST_DIR = os.path.join(DATA, "test_images")

# external generalization set (optional)
SHIPRS_IMG = os.path.join(DATA, "shiprs_val_unseen", "images", "val")
SHIPRS_LBL = os.path.join(DATA, "shiprs_val_unseen", "labels", "val")

BEST = os.path.join(WEIGHTS_DIR, "best.pt")
BEST_FT = os.path.join(WEIGHTS_DIR, "best_finetuned.pt")

# Ultralytics dataset spec used by train / val
DATA_YAML = os.path.join(DATA, "data.yaml")


def _add_pkg_to_syspath():
    """Call from a sub-folder script so `import paths` works."""
    import sys
    if PKG not in sys.path:
        sys.path.insert(0, PKG)
