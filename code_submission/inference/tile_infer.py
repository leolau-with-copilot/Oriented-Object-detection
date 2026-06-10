# 切割式预测
import os, sys, glob, math, cv2, numpy as np
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))
import paths

VAL_IMG = paths.IMG_VAL
VAL_LBL = paths.LBL_VAL
EXTS = ('.jpg', '.jpeg', '.png')

TILE = 640            # tile size in px
OVERLAP = 0.30        # fractional overlap between tiles
CONF = 0.10           # detection confidence threshold
IOU_NMS = 0.50        # rotated-IoU threshold for merging duplicates
IOU_MATCH = 0.50      # rotated-IoU threshold for GT matching (evaluation)
FULL_IMGSZ = 1024     # imgsz for the whole-image pass
TILE_IMGSZ = 640      # imgsz for each tile (== TILE -> native, no resize)


# --------------------------------------------------------------- geometry utils
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


# ------------------------------------------------------------------- inference
def _collect(result, ox=0, oy=0):
    out = []
    if result.obb is None or len(result.obb) == 0:
        return out
    xywhr = result.obb.xywhr.cpu().numpy()       # (N,5): cx,cy,w,h,radians (tile coords)
    confs = result.obb.conf.cpu().numpy()
    for (cx, cy, w, h, r), cf in zip(xywhr, confs):
        cx, cy = cx + ox, cy + oy                # offset back to full-image coords
        out.append(dict(cx=cx, cy=cy, w=w, h=h, r=float(r), conf=float(cf),
                        poly=xywhr_to_poly(cx, cy, w, h, r)))
    return out


def predict_plain(model, img):
    r = model.predict(img, imgsz=FULL_IMGSZ, conf=CONF, verbose=False)[0]
    return rotated_nms(_collect(r))


def predict_tiled(model, img):
    H, W = img.shape[:2]
    dets = _collect(model.predict(img, imgsz=FULL_IMGSZ, conf=CONF, verbose=False)[0])  # full pass
    step = int(TILE * (1 - OVERLAP))
    xs = list(range(0, max(1, W - TILE + 1), step)) + [max(0, W - TILE)]
    ys = list(range(0, max(1, H - TILE + 1), step)) + [max(0, H - TILE)]
    for y0 in sorted(set(ys)):
        for x0 in sorted(set(xs)):
            tile = img[y0:y0 + TILE, x0:x0 + TILE]
            if tile.size == 0:
                continue
            r = model.predict(tile, imgsz=TILE_IMGSZ, conf=CONF, verbose=False)[0]
            dets += _collect(r, ox=x0, oy=y0)
    return rotated_nms(dets)


# ------------------------------------------------------------------- evaluation
def gt_polys(stem, W, H):
    p = os.path.join(VAL_LBL, stem + '.txt')
    out = []
    if os.path.exists(p) and os.path.getsize(p):
        for line in open(p):
            a = line.split()
            if len(a) >= 9:
                out.append(np.array(a[1:9], np.float32).reshape(4, 2) * [W, H])
    return out


def score(detfn, name):
    n_gt = tp = fp = 0
    missed = {}
    for ip in sorted(glob.glob(os.path.join(VAL_IMG, '*'))):
        if not ip.lower().endswith(EXTS):
            continue
        stem = os.path.splitext(os.path.basename(ip))[0]
        img = cv2.imread(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        gts = gt_polys(stem, W, H)
        dets = detfn(img)
        used = set()
        for gi, g in enumerate(gts):
            n_gt += 1
            best, bj = 0.0, -1
            for j, d in enumerate(dets):
                v = poly_iou(g, d['poly'])
                if v > best:
                    best, bj = v, j
            if best >= IOU_MATCH and bj not in used:
                tp += 1; used.add(bj)
            else:
                missed.setdefault(stem, []).append(gi)
        fp += max(0, len(dets) - len(used))
    rec = tp / max(n_gt, 1); prec = tp / max(tp + fp, 1)
    print(f'  {name:6s}: recall {rec:.3f}  precision {prec:.3f}  '
          f'missed {sum(len(v) for v in missed.values())}/{n_gt}')
    return rec, prec, missed


def evaluate(weights):
    model = YOLO(weights)
    print(f'Evaluating on {VAL_IMG}  (tile={TILE}, overlap={OVERLAP}, conf={CONF})')
    _, _, m_plain = score(lambda im: predict_plain(model, im), 'plain')
    _, _, m_tiled = score(lambda im: predict_tiled(model, im), 'tiled')
    plain_set = {(s, gi) for s, gis in m_plain.items() for gi in gis}
    tiled_set = {(s, gi) for s, gis in m_tiled.items() for gi in gis}
    recovered = len(plain_set - tiled_set)     # missed by plain, caught by tiling
    newly = len(tiled_set - plain_set)         # caught by plain, missed by tiling
    print(f'\n  tiling vs plain: recovered {recovered} of plain\'s misses, '
          f'introduced {newly} new misses')
    print('  -> tiling helps' if recovered > newly else
          '  -> tiling does NOT help (keep plain)')


# ------------------------------------------------------------------- submission
def submit(weights, test_dir):
    model = YOLO(weights)
    rows = []
    for ip in sorted(glob.glob(os.path.join(test_dir, '*'))):
        if not ip.lower().endswith(EXTS):
            continue
        stem = os.path.splitext(os.path.basename(ip))[0]
        img = cv2.imread(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        dets = predict_tiled(model, img)
        parts = [f'{d["conf"]:.4f} {d["cx"]/W:.6f} {d["cy"]/H:.6f} '
                 f'{d["w"]/W:.6f} {d["h"]/H:.6f} {d["r"]:.6f}' for d in dets]
        rows.append((stem, ', '.join(parts)))
    out = os.path.join(ROOT, 'submission_tiled.csv')
    with open(out, 'w') as f:
        f.write('Id,result\n')
        for sid, res in rows:
            f.write(f'{sid},{res}\n')
    print(f'wrote {out}  ({len(rows)} images)')


if __name__ == '__main__':
    weights = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') \
        else paths.BEST
    if '--submit' in sys.argv:
        test_dir = sys.argv[sys.argv.index('--submit') + 1]
        submit(weights, test_dir)
    else:
        evaluate(weights)
