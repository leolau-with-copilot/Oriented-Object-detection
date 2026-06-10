# 在自建的验证集上做测试
import os, sys
import yaml
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

ROOT = paths.DATA
DY = paths.DATA_YAML

# pin data.yaml path so val finds images/labels wherever this runs
os.makedirs(paths.DATA, exist_ok=True)
yaml.safe_dump({'path': paths.DATA, 'train': 'images/train', 'val': 'images/val',
                'names': {0: 'ship'}}, open(DY, 'w'), sort_keys=False)

m = YOLO(paths.BEST)

# 1) proper oriented-box metrics on the val split
metrics = m.val(data=DY, split='val', imgsz=1024, conf=0.001, iou=0.6,
                project='val_eval', name='metrics', plots=True, verbose=False)
b = metrics.box
print('\n===== OUR val split (oriented-box eval, in-distribution) =====')
print(f'images        : {len(os.listdir(os.path.join(ROOT, "images", "val")))}')
print(f'mAP@0.50      : {b.map50:.4f}')
print(f'mAP@0.50-0.95 : {b.map:.4f}')
print(f'mAP@0.75      : {b.map75:.4f}')
print(f'precision     : {b.mp:.4f}')
print(f'recall        : {b.mr:.4f}')
print('==============================================================')

# 2) annotated predictions (drawn oriented boxes) for visual review
m.predict(source=os.path.join(ROOT, 'images', 'val'), conf=0.25, imgsz=1024,
          save=True, project='val_eval', name='preds', verbose=False)
print('saved annotated predictions to val_eval/preds/')
