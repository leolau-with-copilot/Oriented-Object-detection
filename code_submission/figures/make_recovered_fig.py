"""Regenerate figures/before_after.png to show ONLY ships that the baseline MISSED
but the fine-tuned model CAUGHT (the recovery cases), limited to 4 before/after pairs.
"""
import os, glob, cv2, numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO

VAL_IMG, VAL_LBL = paths.IMG_VAL, paths.LBL_VAL
IMGSZ, CONF, IOU = 1024, 0.10, 0.5
EXTS = ('.jpg', '.jpeg', '.png')


def gt_polys(stem, W, H):
    p = os.path.join(VAL_LBL, stem + '.txt'); out = []
    if os.path.exists(p) and os.path.getsize(p):
        for l in open(p):
            a = l.split()
            if len(a) >= 9:
                out.append(np.array(a[1:9], np.float32).reshape(4, 2) * [W, H])
    return out


def poly_iou(p1, p2):
    p1, p2 = p1.astype(np.float32), p2.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(p1, p2)
    u = cv2.contourArea(p1) + cv2.contourArea(p2) - inter
    return inter / u if u > 0 else 0.0


def preds(model, ip):
    r = model.predict(ip, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    return (r.obb.xyxyxyxy.cpu().numpy() if (r.obb is not None and len(r.obb))
            else np.zeros((0, 4, 2), np.float32))


def hit(g, P):
    return any(poly_iou(g, p) >= IOU for p in P)


mb, mf = YOLO(paths.BEST), YOLO(paths.BEST_FT)
recovered = []
for ip in sorted(glob.glob(os.path.join(VAL_IMG, '*'))):
    if not ip.lower().endswith(EXTS):
        continue
    stem = os.path.splitext(os.path.basename(ip))[0]
    img = cv2.imread(ip)
    if img is None:
        continue
    H, W = img.shape[:2]
    gts = gt_polys(stem, W, H)
    if not gts:
        continue
    Pb, Pf = preds(mb, ip), preds(mf, ip)
    for g in gts:
        if (not hit(g, Pb)) and hit(g, Pf):          # baseline missed, fine-tune caught
            recovered.append((ip, g, Pb, Pf))
    if len(recovered) >= 4:
        break

recovered = recovered[:4]
print(f'recovered cases shown: {len(recovered)}')

fig, axes = plt.subplots(len(recovered), 2, figsize=(7, 3.2 * len(recovered)), squeeze=False)
for ri, (ip, g, Pb, Pf) in enumerate(recovered):
    img = cv2.imread(ip); H, W = img.shape[:2]
    x, y, w, h = cv2.boundingRect(g.astype(np.float32))
    pad = int(max(w, h) * 1.2) + 14
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    for ci, P in enumerate([Pb, Pf]):
        crop = img[y0:y1, x0:x1].copy()
        cv2.polylines(crop, [(g - [x0, y0]).astype(np.int32)], True, (60, 220, 60), 2)   # GT green
        for p in P:                                                                       # preds blue
            pp = (p - [x0, y0]).astype(np.int32)
            if (pp[:, 0].max() >= 0 and pp[:, 1].max() >= 0 and
                    pp[:, 0].min() <= crop.shape[1] and pp[:, 1].min() <= crop.shape[0]):
                cv2.polylines(crop, [pp], True, (235, 80, 40), 2)
        ax = axes[ri][ci]
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)); ax.axis('off')
        if ri == 0:
            ax.set_title('baseline (missed)' if ci == 0 else 'fine-tuned (detected)', fontsize=11)
fig.suptitle('Ships recovered by fine-tuning (green = GT, blue = prediction)', fontsize=12)
plt.tight_layout()
plt.savefig('figures/before_after.png', dpi=140, bbox_inches='tight')
print('saved figures/before_after.png')
