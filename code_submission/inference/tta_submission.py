# 增强式预测
import os, sys, glob, math, cv2, numpy as np
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))
import paths

WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else paths.BEST_FT
TEST_DIR = sys.argv[2] if len(sys.argv) > 2 else paths.TEST_DIR
CONF = 0.05            # match predict_submission.py
IOU_NMS = 0.50         # rotated-IoU threshold for merging the views
KEEP_EXT = True        # id includes file extension (match predict_submission.py)
SCALES = [1024, 1280]  # multi-scale TTA: predict each view at each imgsz
EXTS = ('.jpg', '.jpeg', '.png', '.JPG')


def wrap_theta(t):
    """Fold any angle into [-pi/2, pi/2) — box-preserving (rect is pi-periodic)."""
    return (t + math.pi / 2) % math.pi - math.pi / 2


def xywhr_to_poly(cx, cy, w, h, r):
    c, s = math.cos(r), math.sin(r)
    R = np.array([[c, -s], [s, c]])
    loc = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    return (loc @ R.T) + np.array([cx, cy])


def poly_iou(p1, p2):
    p1, p2 = p1.astype(np.float32), p2.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(p1, p2)
    u = cv2.contourArea(p1) + cv2.contourArea(p2) - inter
    return inter / u if u > 0 else 0.0


def rotated_nms(dets, iou_thr=IOU_NMS):
    dets = sorted(dets, key=lambda d: -d['conf'])
    keep, sup = [], [False] * len(dets)
    for i in range(len(dets)):
        if sup[i]:
            continue
        keep.append(dets[i])
        for j in range(i + 1, len(dets)):
            if not sup[j] and poly_iou(dets[i]['poly'], dets[j]['poly']) > iou_thr:
                sup[j] = True
    return keep


# un-flip transforms: map a box detected in the FLIPPED frame back to the original.
# (only sign flips / mirrors -> no w/h swap, safe angle math)
def t_ident(cx, cy, w, h, r, W, H):  return cx, cy, w, h, r
def t_hflip(cx, cy, w, h, r, W, H):  return W - cx, cy, w, h, -r          # mirror x, negate angle
def t_vflip(cx, cy, w, h, r, W, H):  return cx, H - cy, w, h, -r          # mirror y, negate angle
def t_rot180(cx, cy, w, h, r, W, H): return W - cx, H - cy, w, h, r       # h+v: angle unchanged

# (image-transform, box-un-transform) pairs
VIEWS = [
    (lambda im: im,                                  t_ident),
    (lambda im: np.ascontiguousarray(im[:, ::-1]),   t_hflip),
    (lambda im: np.ascontiguousarray(im[::-1, :]),   t_vflip),
    (lambda im: np.ascontiguousarray(im[::-1, ::-1]), t_rot180),
]


def collect(result, transform, W, H):
    out = []
    if result.obb is None or len(result.obb) == 0:
        return out
    xywhr = result.obb.xywhr.cpu().numpy()
    confs = result.obb.conf.cpu().numpy()
    for (cx, cy, w, h, r), cf in zip(xywhr, confs):
        cx, cy, w, h, r = transform(cx, cy, w, h, r, W, H)
        r = wrap_theta(r)
        out.append(dict(cx=cx, cy=cy, w=w, h=h, r=r, conf=float(cf),
                        poly=xywhr_to_poly(cx, cy, w, h, r)))
    return out


def main():
    model = YOLO(WEIGHTS)
    print('using weights:', WEIGHTS)
    paths = sorted(p for e in ('*.jpg', '*.jpeg', '*.png', '*.JPG')
                   for p in glob.glob(os.path.join(TEST_DIR, e)))
    if not paths:
        sys.exit(f'no images found in {TEST_DIR}')

    rows, n_det = [], 0
    for k, p in enumerate(paths):
        name = os.path.basename(p)
        sid = name if KEEP_EXT else os.path.splitext(name)[0]
        img = cv2.imread(p)
        if img is None:
            print(f'(skip unreadable) {name}')
            rows.append((sid, ''))
            continue
        H, W = img.shape[:2]
        dets = []
        for make_view, untransform in VIEWS:
            view = make_view(img)
            for s in SCALES:
                r = model.predict(view, imgsz=s, conf=CONF, verbose=False)[0]
                dets += collect(r, untransform, W, H)
        dets = rotated_nms(dets)

        boxes = [f'{d["conf"]:.4f} {d["cx"]/W:.6f} {d["cy"]/H:.6f} '
                 f'{d["w"]/W:.6f} {d["h"]/H:.6f} {d["r"]:.6f}'
                 for d in sorted(dets, key=lambda z: -z['conf'])]
        rows.append((sid, ', '.join(boxes)))
        if boxes:
            n_det += 1
        if (k + 1) % 50 == 0:
            print(f'  {k+1}/{len(paths)}')

    out = os.path.join(ROOT, 'submission_tta.csv')
    with open(out, 'w') as f:
        f.write('Id,result\n')
        for sid, res in rows:
            f.write(f'{sid},{res}\n')
    print(f'wrote {out}  ({len(rows)} images, {n_det} with detections)')


if __name__ == '__main__':
    main()
