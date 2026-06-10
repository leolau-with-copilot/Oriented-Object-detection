"""Train YOLO-OBB on the ship dataset (run on the GPU box).

    conda activate <your-env>      # the env with ultralytics + cu124 torch
    python train_yolo_obb.py

The four augmentations from the plan are mapped to Ultralytics flags below:
  (1) continuous rotation -> degrees=180   (random rotation in [-180,180], OBB re-derived)
  (3) scale jitter        -> scale=0.5 + multi_scale=True + mosaic
  (4) photometric jitter  -> hsv_h / hsv_s / hsv_v
  (+) flips               -> fliplr / flipud
  (2) copy-paste          -> DEFERRED (needs instance masks; OBB copy_paste is mask-based)
"""
import os, sys
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

DATA = paths.DATA_YAML       # written by prepare_yolo_obb.py under the data root

# (re)write data.yaml so its `path` points at the data root wherever this is run
import yaml
os.makedirs(paths.DATA, exist_ok=True)
yaml.safe_dump({'path': paths.DATA, 'train': 'images/train', 'val': 'images/val',
                'names': {0: 'ship'}}, open(DATA, 'w'), sort_keys=False)

if __name__ == '__main__':
    # FROM SCRATCH (no pretrained weights): load the architecture .yaml, not a .pt.
    # The .yaml ships inside the ultralytics package -> no download needed.
    # Smaller model (s) generalizes better than m when training from scratch on a
    # small dataset; bump to yolo11m-obb.yaml only if val mAP plateaus low and the
    # train_batch images show clear underfitting.
    model = YOLO('yolo11s-obb.yaml')   # random init, no pretrained weights
    model.train(
        data=DATA,
        pretrained=False,      # explicit: do NOT load any pretrained backbone
        epochs=300,            # from scratch + tiny data -> needs many epochs
        patience=100,          # converges slowly; give it room before early-stop
        imgsz=1024,
        batch=16,
        device=0,
        save=True,
        plots=True,
        # ---- augmentation: the four tricks ----
        degrees=180.0,         # (1) continuous rotation
        scale=0.5,             # (3) scale jitter (+/- 50%)
        multi_scale=True,      # (3) multi-scale training
        mosaic=1.0,            # (3) multi-scale/context composition
        close_mosaic=20,       # turn mosaic off for the last 20 epochs (clean fine-tune)
        hsv_h=0.015,           # (4) hue
        hsv_s=0.7,             # (4) saturation
        hsv_v=0.4,             # (4) brightness/value
        fliplr=0.5,
        flipud=0.5,
        translate=0.1,
        # copy_paste=0.0,      # (2) leave off until we have instance masks
        project='runs_ship',
        name='yolo11s_obb_scratch',
    )
