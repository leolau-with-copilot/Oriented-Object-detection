# submission.py 的最终预测
import glob
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))
import paths

WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else paths.BEST
TEST_DIR = sys.argv[2] if len(sys.argv) > 2 else paths.TEST_DIR

CONF = 0.05        # 给予高精度模型较低的threshold
KEEP_EXT = True    
VIZ_N = 12         


def resolve_weights(path):
    """Find the trained checkpoint even if Ultralytics nested the run under runs/obb/."""
    candidates = [path, paths.BEST]
    candidates.extend(sorted(
        glob.glob(os.path.join(paths.DATA, 'runs', '**', 'weights', 'best.pt'), recursive=True),
        key=os.path.getmtime,
        reverse=True,
    ))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f'no checkpoint found; tried: {", ".join(seen)}'
    )


def wrap_theta(t):
    """A rectangle is pi-periodic, so map theta into [-pi/2, pi/2) (the GT range)
    without changing the box."""
    return (t + math.pi / 2) % math.pi - math.pi / 2


def corners_px(cx, cy, w, h, theta, W, H):
    """Reconstruct pixel corners from normalized (cx,cy,w,h,theta) — identical
    convention to the GT labels / prepare_yolo_obb.py."""
    cxp, cyp, wp, hp = cx * W, cy * H, w * W, h * H
    dx, dy = wp / 2.0, hp / 2.0
    base = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    c, s = math.cos(theta), math.sin(theta)
    return np.array([(c * x - s * y + cxp, s * x + c * y + cyp) for x, y in base],
                    dtype=np.int32)


def main():
    weights = resolve_weights(WEIGHTS)
    print(f'using weights: {weights}')
    model = YOLO(weights)
    exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG')
    paths = sorted(p for e in exts for p in glob.glob(os.path.join(TEST_DIR, e)))
    if not paths:
        sys.exit(f'no images found in {TEST_DIR}')

    viz_dir = os.path.join(ROOT, 'submission_viz')
    os.makedirs(viz_dir, exist_ok=True)

    ids, results = [], []
    n_det = 0
    for i, p in enumerate(paths):
        name = os.path.basename(p)
        ids.append(name if KEEP_EXT else os.path.splitext(name)[0])

        r = model.predict(p, conf=CONF, verbose=False)[0]
        H, W = r.orig_shape                      # (height, width)
        boxes = []
        rows = []
        if r.obb is not None and len(r.obb) > 0:
            xywhr = r.obb.xywhr.cpu().numpy()    # cx,cy,w,h,theta(rad) in pixels
            confs = r.obb.conf.cpu().numpy()
            for (cx, cy, w, h, th), c in sorted(zip(xywhr, confs),
                                                key=lambda z: -z[1]):
                ncx, ncy, nw, nh = cx / W, cy / H, w / W, h / H
                nth = wrap_theta(float(th))
                boxes.append(f'{c:.4f} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f} {nth:.6f}')
                rows.append((ncx, ncy, nw, nh, nth))
        results.append(', '.join(boxes))         # "" if no detections
        if boxes:
            n_det += 1

        # convention sanity-check overlay (from the SUBMITTED normalized values)
        if i < VIZ_N:
            img = cv2.imread(p)
            for ncx, ncy, nw, nh, nth in rows:
                cv2.polylines(img, [corners_px(ncx, ncy, nw, nh, nth, W, H)],
                              True, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(viz_dir, name), img)

    pd.DataFrame({'Id': ids, 'result': results}).to_csv(
        os.path.join(ROOT, 'submission.csv'), index=False)
    print(f'wrote submission.csv: {len(ids)} images, {n_det} with detections')
    print(f'wrote {min(VIZ_N, len(paths))} overlays to submission_viz/ '
          f'(open them to confirm boxes hug ships)')


if __name__ == '__main__':
    main()
