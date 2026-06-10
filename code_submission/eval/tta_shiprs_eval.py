# 验证tta的有效性
import os, sys, glob, math, cv2, numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

IMG = paths.SHIPRS_IMG
LBL = paths.SHIPRS_LBL
WEIGHTS = paths.BEST_FT
CONF = 0.001                 # low -> full PR curve for AP
SCALES = [1024, 1280]
EXTS = ('.bmp', '.jpg', '.jpeg', '.png')


def wrap_theta(t): return (t + math.pi/2) % math.pi - math.pi/2
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
def rotated_nms(dets, thr=0.5):
    dets = sorted(dets, key=lambda d: -d['conf']); keep, sup = [], [False]*len(dets)
    for i in range(len(dets)):
        if sup[i]: continue
        keep.append(dets[i])
        for j in range(i+1, len(dets)):
            if not sup[j] and poly_iou(dets[i]['poly'], dets[j]['poly']) > thr: sup[j] = True
    return keep

def t_ident(cx, cy, w, h, r, W, H): return cx, cy, w, h, r
def t_h(cx, cy, w, h, r, W, H):     return W-cx, cy, w, h, -r
def t_v(cx, cy, w, h, r, W, H):     return cx, H-cy, w, h, -r
def t_180(cx, cy, w, h, r, W, H):   return W-cx, H-cy, w, h, r
VIEWS = [(lambda im: im, t_ident),
         (lambda im: np.ascontiguousarray(im[:, ::-1]), t_h),
         (lambda im: np.ascontiguousarray(im[::-1, :]), t_v),
         (lambda im: np.ascontiguousarray(im[::-1, ::-1]), t_180)]

def collect(res, tr, W, H):
    out = []
    if res.obb is None or len(res.obb) == 0: return out
    for (cx, cy, w, h, r), cf in zip(res.obb.xywhr.cpu().numpy(), res.obb.conf.cpu().numpy()):
        cx, cy, w, h, r = tr(cx, cy, w, h, r, W, H); r = wrap_theta(r)
        out.append(dict(conf=float(cf), poly=xywhr_to_poly(cx, cy, w, h, r)))
    return out

def gt_polys(stem, W, H):
    p = os.path.join(LBL, stem + '.txt'); out = []
    if os.path.exists(p) and os.path.getsize(p):
        for l in open(p):
            a = l.split()
            if len(a) >= 9: out.append(np.array(a[1:9], np.float32).reshape(4, 2) * [W, H])
    return out

def voc_ap(rec, prec):
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre)-2, -1, -1): mpre[i] = max(mpre[i], mpre[i+1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]))

def ap_from(records, npos):
    records.sort(key=lambda z: -z[0])
    tp = np.array([r[1] for r in records]); fp = 1 - tp
    tpc, fpc = np.cumsum(tp), np.cumsum(fp)
    rec = tpc / max(npos, 1); prec = tpc / np.maximum(tpc + fpc, 1e-9)
    return voc_ap(rec, prec), (tpc[-1]/npos if len(tpc) else 0.0)

model = YOLO(WEIGHTS)
rec_plain, rec_tta, npos = [], [], 0
paths = sorted(p for p in glob.glob(os.path.join(IMG, '*')) if p.lower().endswith(EXTS))
for k, ip in enumerate(paths):
    stem = os.path.splitext(os.path.basename(ip))[0]
    img = cv2.imread(ip)
    if img is None: continue
    H, W = img.shape[:2]
    gts = gt_polys(stem, W, H); npos += len(gts)

    tta_dets, plain_dets = [], []
    for vi, (mk, tr) in enumerate(VIEWS):
        view = mk(img)
        for s in SCALES:
            d = collect(model.predict(view, imgsz=s, conf=CONF, verbose=False)[0], tr, W, H)
            tta_dets += d
            if vi == 0 and s == 1024:
                plain_dets += d                  # single identity@1024 pass = no-TTA
    for tag, dets, store in [('plain', rotated_nms(plain_dets), rec_plain),
                             ('tta',   rotated_nms(tta_dets),   rec_tta)]:
        dets = sorted(dets, key=lambda d: -d['conf']); matched = [False]*len(gts)
        for d in dets:
            best, bj = 0.0, -1
            for gi, g in enumerate(gts):
                if matched[gi]: continue
                v = poly_iou(g, d['poly'])
                if v > best: best, bj = v, gi
            tp = 1 if (best >= 0.5 and bj >= 0) else 0
            if tp: matched[bj] = True
            store.append((d['conf'], tp))
    if (k+1) % 80 == 0: print(f'  {k+1}/{len(paths)}', flush=True)

ap_p, r_p = ap_from(rec_plain, npos)
ap_t, r_t = ap_from(rec_tta, npos)
print('\n===== fine-tuned on ShipRSImageNet unseen (our AP@0.5 evaluator) =====')
print(f'  plain (1 view) : mAP50 {ap_p:.4f}   recall {r_p:.4f}')
print(f'  +OBB-TTA       : mAP50 {ap_t:.4f}   recall {r_t:.4f}   (delta {ap_t-ap_p:+.4f})')
