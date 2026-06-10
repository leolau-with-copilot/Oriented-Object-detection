# 将模型在前500张图片上进行评估
import os
import sys
import glob
import xml.etree.ElementTree as ET

import yaml
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

VOC = os.path.join(paths.DATA, 'ShipRSImageNet', 'ShipRSImageNet_V1', 'VOC_Format')
JPEG = os.path.join(VOC, 'JPEGImages')
ANNO = os.path.join(VOC, 'Annotations')
VALLIST = os.path.join(VOC, 'ImageSets', 'val.txt')
OUT = os.path.join(paths.DATA, 'shiprs_val_unseen')


def clip01(v):
    return min(max(v, 0.0), 1.0)


def poly_label(xml_path):
    """Return list of normalized 'x1 y1 ... x4 y4' strings for an annotation."""
    root = ET.parse(xml_path).getroot()
    W = float(root.findtext('size/width'))
    H = float(root.findtext('size/height'))
    out = []
    for obj in root.findall('object'):
        p = obj.find('polygon')
        if p is not None:
            xs = [float(p.findtext(f'x{i}')) for i in range(1, 5)]
            ys = [float(p.findtext(f'y{i}')) for i in range(1, 5)]
        else:                       # fall back to rotated_box -> corners
            import math
            rb = obj.find('rotated_box')
            if rb is None:
                continue
            cx = float(rb.findtext('cx')); cy = float(rb.findtext('cy'))
            w = float(rb.findtext('width')); h = float(rb.findtext('height'))
            a = float(rb.findtext('rot'))
            c, s = math.cos(a), math.sin(a)
            xs, ys = [], []
            for dx, dy in [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]:
                xs.append(cx + c*dx - s*dy); ys.append(cy + s*dx + c*dy)
        coords = []
        for x, y in zip(xs, ys):
            coords += [f'{clip01(x/W):.6f}', f'{clip01(y/H):.6f}']
        out.append('0 ' + ' '.join(coords))
    return out


def main():
    ids = [l.strip() for l in open(VALLIST) if l.strip()]
    img_dir = os.path.join(OUT, 'images', 'val')
    lbl_dir = os.path.join(OUT, 'labels', 'val')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # exclude any image the model saw during training (same source images -> leakage)
    our_train = {os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob('ship_yolo_obb/images/train/*')}
    kept = skipped = n_box = 0
    for fn in ids:
        stem = os.path.splitext(fn)[0]
        if stem in our_train:
            skipped += 1
            continue                          # UNSEEN-only: drop training overlaps
        src_img = os.path.join(JPEG, stem + '.bmp')
        xml = os.path.join(ANNO, stem + '.xml')
        if not (os.path.exists(src_img) and os.path.exists(xml)):
            continue
        link = os.path.join(img_dir, stem + '.bmp')
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.abspath(src_img), link)
        lines = poly_label(xml)
        n_box += len(lines)
        with open(os.path.join(lbl_dir, stem + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        kept += 1

    with open(os.path.join(OUT, 'data.yaml'), 'w') as f:
        yaml.safe_dump({'path': OUT, 'train': 'images/val', 'val': 'images/val',
                        'names': {0: 'ship'}}, f, sort_keys=False)

    print(f'UNSEEN val images kept: {kept}  ({skipped} excluded as training overlap), '
          f'{n_box} boxes')

    m = YOLO(paths.BEST)
    r = m.val(data=os.path.join(OUT, 'data.yaml'), split='val', imgsz=1024,
              conf=0.001, iou=0.6, project='val_eval', name='shiprs_unseen',
              plots=False, verbose=False)
    b = r.box
    print('\n===== ShipRSImageNet val (UNSEEN only, oriented-box eval) =====')
    print(f'mAP@0.50      : {b.map50:.4f}')
    print(f'mAP@0.75      : {b.map75:.4f}')
    print(f'mAP@0.50-0.95 : {b.map:.4f}')
    print(f'precision     : {b.mp:.4f}')
    print(f'recall        : {b.mr:.4f}')
    print('=============================================================')


if __name__ == '__main__':
    main()
