"""Find the ships our model MISSES on our own val split and characterize why.
Runs best.pt on the val split, matches predictions to GT by rotated
IoU, collects the unmatched GT (false negatives), prints a size/density/contrast
breakdown, and saves a montage of the missed ships into figures/missed_ships.png .
"""
import os, glob, cv2, numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO

IMG_DIR = paths.IMG_VAL
LBL_DIR = paths.LBL_VAL
CONF = 0.10            # low conf -> a "miss" means the model found nothing even when lenient
IOU_MATCH = 0.5


def gt_polys(stem, W, H):
    p = os.path.join(LBL_DIR, stem + '.txt')
    out = []
    if not os.path.exists(p):
        return out
    for line in open(p):
        a = line.split()
        if len(a) >= 9:
            xy = np.array(a[1:9], np.float32).reshape(4, 2) * np.array([W, H], np.float32)
            out.append(xy)
    return out


def poly_iou(p1, p2):
    p1 = p1.astype(np.float32); p2 = p2.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(p1, p2)
    a1, a2 = cv2.contourArea(p1), cv2.contourArea(p2)
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0


model = YOLO(paths.BEST)
missed = []
n_gt = n_hit = 0
imgs = sorted(glob.glob(os.path.join(IMG_DIR, '*')))
for ip in imgs:
    stem = os.path.splitext(os.path.basename(ip))[0]
    img = cv2.imread(ip)
    if img is None:
        continue
    H, W = img.shape[:2]
    gts = gt_polys(stem, W, H)
    if not gts:
        continue
    r = model.predict(ip, imgsz=1024, conf=CONF, verbose=False)[0]
    preds = (r.obb.xyxyxyxy.cpu().numpy() if (r.obb is not None and len(r.obb)) else
             np.zeros((0, 4, 2), np.float32))
    centers = np.array([g.mean(0) for g in gts])
    for gi, g in enumerate(gts):
        n_gt += 1
        best = max((poly_iou(g, p) for p in preds), default=0.0)
        if best >= IOU_MATCH:
            n_hit += 1
            continue
        # ----- characterize this miss -----
        area = cv2.contourArea(g.astype(np.float32))
        x, y, w, h = cv2.boundingRect(g.astype(np.float32))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(mask, g.astype(np.int32), 255)
        inten = float(gray[mask > 0].mean()) if (mask > 0).any() else 0.0
        diag = np.hypot(w, h)
        neigh = int(((np.hypot(*(centers - centers[gi]).T) < 1.5 * diag).sum()) - 1)
        missed.append(dict(stem=stem, ip=ip, poly=g, area=area, w=w, h=h,
                           inten=inten, neigh=neigh, best_iou=best))

recall = n_hit / max(n_gt, 1)
print(f'GT boxes: {n_gt},  matched: {n_hit},  MISSED: {len(missed)}  (recall={recall:.3f})')

ar = np.array([m['area'] for m in missed])
print('\n--- why missed: size breakdown ---')
for lo, hi, lab in [(0, 1500, 'tiny  (<1500 px^2)'), (1500, 4000, 'small (1500-4000)'),
                    (4000, 1e9, 'large (>4000)')]:
    sel = (ar >= lo) & (ar < hi)
    print(f'  {lab:20s}: {sel.sum():3d}  ({100*sel.mean():.0f}%)')
dense = sum(1 for m in missed if m['neigh'] >= 2)
dark = sum(1 for m in missed if m['inten'] < 80)
print(f'  in dense cluster (>=2 neighbors): {dense} ({100*dense/len(missed):.0f}%)')
print(f'  low-brightness (<80/255)        : {dark} ({100*dark/len(missed):.0f}%)')

# montage: smallest/most-representative misses, cropped with context
missed.sort(key=lambda m: m['area'])
show = missed[:9]
cols = 3; rows = int(np.ceil(len(show) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.4 * rows))
axes = np.array(axes).ravel()
for ax, m in zip(axes, show):
    img = cv2.imread(m['ip']); H, W = img.shape[:2]
    x, y, w, h = cv2.boundingRect(m['poly'].astype(np.float32))
    pad = int(max(w, h) * 0.9) + 10
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    crop = img[y0:y1, x0:x1].copy()
    poly = (m['poly'] - [x0, y0]).astype(np.int32)
    cv2.polylines(crop, [poly], True, (40, 40, 235), 2, cv2.LINE_AA)   # red = missed GT
    ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    ax.set_title(f"area={m['area']:.0f}px  neigh={m['neigh']}\nbright={m['inten']:.0f}  maxIoU={m['best_iou']:.2f}",
                 fontsize=9)
    ax.axis('off')
for ax in axes[len(show):]:
    ax.axis('off')
fig.suptitle('Ships MISSED on our val split (red = un-detected GT) — mostly tiny / dense / low-contrast',
             fontsize=12)
plt.tight_layout()
plt.savefig('figures/missed_ships.png', dpi=130, bbox_inches='tight')
plt.close()
print('\nsaved figures/missed_ships.png')
