# 原始数据格式转换（按照yolo格式）
import math
import os
import shutil
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

SRC_IMG = paths.RAW_IMG      # raw competition images (data/raw/images/train)
SRC_LBL = paths.RAW_LBL      # raw "cx cy w h theta" labels (data/raw/labels/train)
OUT = paths.DATA             # writes images/{train,val}, labels/{train,val}, data.yaml here
VAL_FRACTION = 0.15
SEED = 42


def corners_norm(cx, cy, w, h, theta, W, H):
    
    cxp, cyp, wp, hp = cx * W, cy * H, w * W, h * H
    dx, dy = wp / 2.0, hp / 2.0
    base = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    c, s = math.cos(theta), math.sin(theta)
    pts = []
    for x, y in base:
        px = (c * x - s * y + cxp) / W
        py = (s * x + c * y + cyp) / H
        pts.append((min(max(px, 0.0), 1.0), min(max(py, 0.0), 1.0)))  # clip to frame
    return pts


def read_boxes(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                boxes.append(tuple(map(float, p[:5])))
    return boxes


def main():
    import random
    stems = sorted(
        os.path.splitext(f)[0] for f in os.listdir(SRC_IMG)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        and os.path.exists(os.path.join(SRC_LBL, os.path.splitext(f)[0] + '.txt'))
    )
    random.Random(SEED).shuffle(stems)
    n_val = round(len(stems) * VAL_FRACTION)
    splits = {'val': stems[:n_val], 'train': stems[n_val:]}

    for split, names in splits.items():
        img_dir = os.path.join(OUT, 'images', split)
        lbl_dir = os.path.join(OUT, 'labels', split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        n_box = 0
        for stem in names:
            img_name = stem + '.jpg'
            shutil.copy2(os.path.join(SRC_IMG, img_name), os.path.join(img_dir, img_name))
            lines = []
            W, H = Image.open(os.path.join(img_dir, img_name)).size
            for cx, cy, w, h, theta in read_boxes(os.path.join(SRC_LBL, stem + '.txt')):
                pts = corners_norm(cx, cy, w, h, theta, W, H)
                coords = ' '.join(f'{x:.6f} {y:.6f}' for x, y in pts)
                lines.append(f'0 {coords}')
                n_box += 1
            with open(os.path.join(lbl_dir, stem + '.txt'), 'w') as f:
                f.write('\n'.join(lines) + ('\n' if lines else ''))
        print(f'{split:5s}: {len(names):4d} images, {n_box:5d} boxes')

    with open(os.path.join(OUT, 'data.yaml'), 'w') as f:
        f.write(
            '# Ultralytics YOLO-OBB dataset. `path` is set automatically by\n'
            '# train_yolo_obb.py to this folder, so you can scp it anywhere.\n'
            'path: .\n'
            'train: images/train\n'
            'val: images/val\n'
            'names:\n'
            '  0: ship\n'
        )
    print(f'\nwrote {OUT}/data.yaml')
    print(f'total labelled images: {len(stems)}  (seed={SEED}, val={VAL_FRACTION:.0%})')


if __name__ == '__main__':
    main()
