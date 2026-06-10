# 数据增强代码
import os, math, cv2, numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMG_DIR, LBL_DIR = paths.IMG_TRAIN, paths.RAW_LBL
EXTS = ('.jpg', '.jpeg', '.png')
GREEN = (60, 220, 60)


def load_boxes(stem):
    """Return [(cx, cy, w, h, theta), ...] from a 5-value label file."""
    p = os.path.join(LBL_DIR, stem + '.txt')
    out = []
    if os.path.exists(p):
        for l in open(p):
            a = l.split()
            if not a:
                continue
            if len(a) != 5:
                raise ValueError(
                    f'{p}: expected 5-value "cx cy w h theta" labels, got {len(a)} '
                    f'values. Point LBL_DIR at the raw labels (data/raw/labels).')
            out.append(tuple(map(float, a)))
    return out


def img_path(stem):
    for e in EXTS:
        p = os.path.join(IMG_DIR, stem + e)
        if os.path.exists(p):
            return p
    return None


def corners_px(box, W, H):
    """(cx,cy,w,h,theta) normalized -> 4x2 pixel corners (le90, y-down, R=[[c,-s],[s,c]])."""
    cx, cy, w, h, th = box
    cxp, cyp, wp, hp = cx * W, cy * H, w * W, h * H
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]])
    loc = np.array([[-wp / 2, -hp / 2], [wp / 2, -hp / 2],
                    [wp / 2, hp / 2], [-wp / 2, hp / 2]])
    return (loc @ R.T) + np.array([cxp, cyp])


def draw(img, polys, color=GREEN, t=None):
    im = img.copy()
    if t is None:
        t = max(2, int(round(min(im.shape[:2]) / 300)))
    for p in polys:
        cv2.polylines(im, [p.astype(np.int32)], True, color, t, cv2.LINE_AA)
    return im


def apply_affine(polys, M):
    out = []
    for p in polys:
        ph = np.hstack([p, np.ones((4, 1))])
        out.append((ph @ M.T)[:, :2])
    return out


# ----------------------------------------------------- pick the clearest image
cands = []
for fn in os.listdir(LBL_DIR):
    stem = os.path.splitext(fn)[0]
    boxes = load_boxes(stem)
    if len(boxes) != 1:
        continue
    p = img_path(stem)
    if not p:
        continue
    im = cv2.imread(p)
    if im is None:
        continue
    H, W = im.shape[:2]
    cx, cy, w, h, th = boxes[0]
    area = w * h
    if area < 0.04 or area > 0.5:           # not too tiny, not whole-frame
        continue
    blur = cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    cands.append((blur, stem, area))
cands.sort(reverse=True)
print('top clear candidates (blur, stem, area):')
for c in cands[:6]:
    print('  ', round(c[0], 0), c[1], round(c[2], 3))

stem = cands[0][1]
im = cv2.imread(img_path(stem))
H, W = im.shape[:2]
polys = [corners_px(b, W, H) for b in load_boxes(stem)]
print(f'\nchosen: {stem}  ({W}x{H})')

# extra images for the mosaic (next clear singles)
extra = [c[1] for c in cands[1:4]]


# --------------------------------------------------------------- augmentations
def aug_rotate(im, polys, deg=57):
    H, W = im.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), deg, 1.0)
    return cv2.warpAffine(im, M, (W, H), borderValue=(114, 114, 114)), apply_affine(polys, M)


def aug_scale(im, polys, s=1.45):
    H, W = im.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
    return cv2.warpAffine(im, M, (W, H), borderValue=(114, 114, 114)), apply_affine(polys, M)


def aug_hsv(im, polys, hg=0.015, sg=0.7, vg=0.4):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    r = np.array([1 + hg, 1 + sg, 1 + vg])          # fixed gains (illustrative)
    hsv[..., 0] = (hsv[..., 0] * r[0]) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), polys


def aug_fliplr(im, polys):
    H, W = im.shape[:2]
    out = im[:, ::-1].copy()
    return out, [np.column_stack([W - 1 - p[:, 0], p[:, 1]]) for p in polys]


def aug_flipud(im, polys):
    H, W = im.shape[:2]
    out = im[::-1].copy()
    return out, [np.column_stack([p[:, 0], H - 1 - p[:, 1]]) for p in polys]


def aug_mosaic(stems, cell=512):
    """Simple mosaic: the chosen image + ONE other, stitched side by side,
    with boxes mapped into each cell."""
    stems = stems[:2]
    canvas = np.full((cell, cell * 2, 3), 114, np.uint8)
    out_polys = []
    offs = [(0, 0), (cell, 0)]
    for (ox, oy), st in zip(offs, stems):
        sim = cv2.imread(img_path(st))
        sh, sw = sim.shape[:2]
        r = cell / max(sh, sw)
        nw, nh = int(sw * r), int(sh * r)
        rim = cv2.resize(sim, (nw, nh))
        canvas[oy:oy + nh, ox:ox + nw] = rim
        for b in load_boxes(st):
            p = corners_px(b, sw, sh) * r + np.array([ox, oy])
            out_polys.append(p)
    return canvas, out_polys


orig = draw(im, polys)
# one SEPARATE before/after figure per transformation (flips omitted)
augs = [
    ('rotation', 'Rotation (degrees=180), here 57 deg', draw(*aug_rotate(im, polys))),
    ('scale',    'Scale jitter (scale=0.5), here x1.45', draw(*aug_scale(im, polys))),
    ('mosaic',   'Mosaic (chosen + one other image)',    draw(*aug_mosaic([stem] + extra))),
    ('hsv',      'Photometric HSV jitter',               draw(*aug_hsv(im, polys))),
]

for key, name, aim in augs:
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    a0.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    a0.set_title('Before (original)', fontsize=11)
    a0.axis('off')
    a1.imshow(cv2.cvtColor(aim, cv2.COLOR_BGR2RGB))
    a1.set_title('After: ' + name, fontsize=11)
    a1.axis('off')
    plt.tight_layout()
    fn = f'figures/aug_{key}.png'
    plt.savefig(fn, dpi=140, bbox_inches='tight')
    plt.close()
    print('saved', fn)
