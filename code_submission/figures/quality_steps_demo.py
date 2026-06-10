"""Illustrate the 3-step evolution of our 'observability' filter and, for each
step, surface CLEARLY-OBSERVABLE ships that the filter would wrongly drop.
This is the evidence behind the decision to keep ALL data.

Step 1  whole-image haze (dark-channel) + blur (Laplacian variance)
Step 2  per-box FFT high-frequency ratio (Hann window, size-normalized) + grad_in
Step 3  inside-vs-'donut' local contrast (grad_in / grad_ratio / inten_z)

Saves three montages into figures/: step1_overdrop.png, step2_overdrop.png,
step3_overdrop.png .
"""
import os, math, cv2, numpy as np, pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# NOTE: needs the ORIGINAL 5-value labels "cx cy w h theta", not the 9-value
# YOLO-OBB polygons under data/labels (those would be misread as boxes).
IMG_DIR, LBL_DIR = paths.IMG_TRAIN, paths.RAW_LBL
EXTS = ('.jpg', '.jpeg', '.png')
os.makedirs('figures', exist_ok=True)


# ----------------------------------------------------------------------- utils
def load_boxes(p):
    """Return [(cx, cy, w, h, theta), ...] from a 5-value label file."""
    out = []
    if not os.path.exists(p):
        return out
    for line in open(p):
        a = line.split()
        if not a:
            continue
        if len(a) != 5:
            raise ValueError(
                f'{p}: expected 5-value "cx cy w h theta" labels, got {len(a)} values. '
                f'Point LBL_DIR at the raw labels (data/raw/labels).')
        out.append(tuple(map(float, a)))
    return out


def extract_scaled_chip(img, cx, cy, w, h, theta, scale=1.5):
    H, W = img.shape[:2]
    cxp, cyp, wp, hp = cx * W, cy * H, w * W, h * H
    M = cv2.getRotationMatrix2D((cxp, cyp), math.degrees(theta), 1.0)
    rot = cv2.warpAffine(img, M, (W, H))
    cw, ch = int(round(wp * scale)), int(round(hp * scale))
    x0, y0 = int(round(cxp - cw / 2)), int(round(cyp - ch / 2))
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + cw), min(H, y0 + ch)
    chip = rot[Y0:Y1, X0:X1]
    if chip.size == 0:
        return None, None
    ih, iw = chip.shape[:2]
    ccx, ccy = cxp - X0, cyp - Y0
    mask = np.zeros((ih, iw), bool)
    ix0, ix1 = max(0, int(round(ccx - wp / 2))), min(iw, int(round(ccx + wp / 2)))
    iy0, iy1 = max(0, int(round(ccy - hp / 2))), min(ih, int(round(ccy + hp / 2)))
    mask[iy0:iy1, ix0:ix1] = True
    return chip, mask


def box_crop(img, cx, cy, w, h, theta):
    """Upright crop of just the box interior (1.0x)."""
    chip, mask = extract_scaled_chip(img, cx, cy, w, h, theta, scale=1.0)
    return chip


def obb_corners(cx, cy, w, h, theta, W, H, scale=1.0):
    """Rotated-rect corners in pixels (le90, y-down R=[[c,-s],[s,c]])."""
    cxp, cyp = cx * W, cy * H
    wp, hp = w * W * scale, h * H * scale
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    loc = np.array([[-wp/2, -hp/2], [wp/2, -hp/2], [wp/2, hp/2], [-wp/2, hp/2]])
    return ((loc @ R.T) + np.array([cxp, cyp])).astype(np.int32)


def draw_boxes(img, cx, cy, w, h, theta, outer=False):
    """Full image with the original (green) rotated box; if `outer`, also the
    1.5x donut (red) box. The donut/outer box belongs ONLY to the step-3 method."""
    im = img.copy()
    H, W = im.shape[:2]
    t = max(2, int(round(min(H, W) / 250)))
    cv2.polylines(im, [obb_corners(cx, cy, w, h, theta, W, H, 1.0)], True, (60, 220, 60), t, cv2.LINE_AA)
    if outer:
        cv2.polylines(im, [obb_corners(cx, cy, w, h, theta, W, H, 1.5)], True, (40, 40, 235), t, cv2.LINE_AA)
    return im


# ---------------------------------------------------------- step-1 whole image
def dark_channel_haze(img, patch=15):
    """Dark-channel-prior haze score in [0,1]; higher = hazier/foggier."""
    mn = img.min(axis=2).astype(np.float32)
    dark = cv2.erode(mn, np.ones((patch, patch), np.uint8))
    return float(dark.mean() / 255.0)


def lap_blur(gray):
    """Variance of Laplacian; lower = blurrier."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ------------------------------------------------------------- step-2 per-box
def fft_highfreq_ratio(gray_chip, size=128, r_frac=0.25):
    """High-frequency energy fraction of a box crop, Hann-windowed and resized
    to a fixed size so the score is decoupled from absolute resolution.
    Lower = blurrier (less high-freq detail)."""
    if gray_chip is None or gray_chip.size == 0:
        return np.nan
    g = cv2.resize(gray_chip.astype(np.float32), (size, size))
    wy = np.hanning(size)[:, None]
    wx = np.hanning(size)[None, :]
    g = g * (wy * wx)                               # Hann window -> kill edge leakage
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(F) ** 2
    cy = cx = size // 2
    yy, xx = np.ogrid[:size, :size]
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high = rad > (r_frac * size / 2)
    tot = mag.sum() + 1e-9
    return float(mag[high].sum() / tot)


def grad_in_of(chip, mask):
    g = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    gmag = np.sqrt(gx * gx + gy * gy)
    inside, donut = mask, ~mask
    if inside.sum() < 9 or donut.sum() < 9:
        return None
    gin = float(gmag[inside].mean()); gout = float(gmag[donut].mean())
    mi, mo, so = g[inside].mean(), g[donut].mean(), g[donut].std()
    return dict(grad_in=gin, grad_out=gout,
                grad_ratio=gin / (gout + 1e-6),
                inten_z=abs(mi - mo) / (so + 1e-6))


# ============================================================ scan everything
img_rows, box_rows = [], []
names = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith(EXTS))
for k, name in enumerate(names):
    stem = os.path.splitext(name)[0]
    boxes = load_boxes(os.path.join(LBL_DIR, stem + '.txt'))
    if not boxes:
        continue
    img = cv2.imread(os.path.join(IMG_DIR, name))
    if img is None:
        continue
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    haze = dark_channel_haze(img)
    blur = lap_blur(gray)
    max_gin = 0.0
    for bi, (cx, cy, w, h, th) in enumerate(boxes):
        chip, mask = extract_scaled_chip(img, cx, cy, w, h, th)
        if chip is None:
            continue
        d = grad_in_of(chip, mask)
        if d is None:
            continue
        crop = box_crop(img, cx, cy, w, h, th)
        fr = fft_highfreq_ratio(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) if crop is not None else np.nan
        max_gin = max(max_gin, d['grad_in'])
        box_rows.append(dict(name=name, box_idx=bi, cx=cx, cy=cy, theta=th,
                             w=w, h=h, W=W, H=H, box_w_px=w * W, box_h_px=h * H,
                             area_px=w * W * h * H, fft_ratio=fr, **d))
    img_rows.append(dict(name=name, haze=haze, blur=blur, max_grad_in=max_gin))
    if (k + 1) % 200 == 0:
        print(f'  scanned {k+1}/{len(names)}')

idf = pd.DataFrame(img_rows)
bdf = pd.DataFrame(box_rows)
print(f'images with ships: {len(idf)},  boxes scored: {len(bdf)}')


# ================================================== figure helper (whole imgs)
def montage_images(sub, title, fname, cols=3, capfn=None):
    sub = sub.reset_index(drop=True)
    n = len(sub); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows))
    axes = np.array(axes).ravel()
    for ax, (_, r) in zip(axes, sub.iterrows()):
        img = cv2.imread(os.path.join(IMG_DIR, r['name']))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(capfn(r) if capfn else r['name'], fontsize=9)
        ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(f'figures/{fname}', dpi=130, bbox_inches='tight')
    plt.close()
    print('saved figures/' + fname)


def montage_full(sub, title, fname, cols=3, capfn=None, outer=False):
    """Full images (with background), each marked by the original (green) box;
    if `outer`, also the 1.5x donut (red) box (step-3 only)."""
    sub = sub.reset_index(drop=True)
    n = len(sub); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows))
    axes = np.array(axes).ravel()
    for ax, (_, r) in zip(axes, sub.iterrows()):
        img = cv2.imread(os.path.join(IMG_DIR, r['name']))
        vis = draw_boxes(img, r['cx'], r['cy'], r['w'], r['h'], r['theta'], outer=outer)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(capfn(r) if capfn else r['name'], fontsize=9)
        ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    legend = '   [green = original box, red = 1.5x donut box]' if outer else '   [green = original box]'
    fig.suptitle(title + legend, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'figures/{fname}', dpi=130, bbox_inches='tight')
    plt.close()
    print('saved figures/' + fname)


def montage_chips(sub, title, fname, cols=3, scale=1.5, capfn=None):
    sub = sub.reset_index(drop=True)
    n = len(sub); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.0 * rows))
    axes = np.array(axes).ravel()
    for ax, (_, r) in zip(axes, sub.iterrows()):
        img = cv2.imread(os.path.join(IMG_DIR, r['name']))
        chip, mask = extract_scaled_chip(img, r['cx'], r['cy'], r['w'], r['h'], r['theta'], scale)
        chip = chip.copy()
        ys, xs = np.where(mask)
        if len(xs):
            cv2.rectangle(chip, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 255, 0), 2)
        ax.imshow(cv2.cvtColor(chip, cv2.COLOR_BGR2RGB))
        ax.set_title(capfn(r) if capfn else r['name'], fontsize=9)
        ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(f'figures/{fname}', dpi=130, bbox_inches='tight')
    plt.close()
    print('saved figures/' + fname)


# =================== STEP 1: whole-image filter wrongly drops sharp-ship scenes
# "drop" rule: haziest decile OR blurriest decile.
haze_thr = idf['haze'].quantile(0.90)
blur_thr = idf['blur'].quantile(0.10)
flagged1 = idf[(idf['haze'] >= haze_thr) | (idf['blur'] <= blur_thr)].copy()
# but these scenes actually contain SHARP ships (high max grad_in) -> good photos
good_drop1 = flagged1.sort_values('max_grad_in', ascending=False).head(6)
montage_images(
    good_drop1,
    'STEP 1 (whole-image haze/blur) — flagged "low quality" yet a sharp ship is plainly visible',
    'step1_overdrop.png',
    capfn=lambda r: f"haze={r['haze']:.2f}  blur={r['blur']:.0f}\n(max grad_in={r['max_grad_in']:.0f})")

# =================== STEP 2: FFT high-freq + grad_in wrongly drops big clear ships
# "drop" rule: lowest-FFT decile (read as 'blurry').
f2 = bdf.dropna(subset=['fft_ratio'])
fft_thr = f2['fft_ratio'].quantile(0.10)
flagged2 = f2[f2['fft_ratio'] <= fft_thr]
# yet these are LARGE, high-contrast ships (high grad_in, large area): low FFT is a
# size artefact, not blur -> good photos the FFT rule drops.
# NOTE: the FFT method scores the box INTERIOR only -> single (original) box, no donut.
good_drop2 = flagged2[flagged2['grad_in'] > f2['grad_in'].median()] \
    .sort_values('area_px', ascending=False).head(6)
montage_full(
    good_drop2,
    'STEP 2 (FFT high-freq + grad_in) — flagged "blurry" but really just large, sharp ships',
    'step2_overdrop.png', outer=False,
    capfn=lambda r: f"fft={r['fft_ratio']:.3f}  grad_in={r['grad_in']:.0f}  area={r['area_px']:.0f}px")

# =================== STEP 3: donut rule still over-flags some observable ships
CUES = ['grad_in', 'grad_ratio', 'inten_z']
thr = {c: bdf[c].quantile(0.10) for c in CUES}
weak = pd.concat([bdf[c] < thr[c] for c in CUES], axis=1).all(axis=1)
flagged3 = bdf[weak].copy()
# pick the borderline ones with the HIGHEST grad_in among the flagged: visibly a ship
good_drop3 = flagged3.sort_values('grad_in', ascending=False).head(6)
montage_full(
    good_drop3,
    'STEP 3 (inside-vs-donut) — even the final rule flags ships still discernible to the eye',
    'step3_overdrop.png', outer=True,
    capfn=lambda r: f"grad_in={r['grad_in']:.0f}  ratio={r['grad_ratio']:.2f}  inten_z={r['inten_z']:.2f}")

print('\nDONE. thresholds:',
      f'haze>={haze_thr:.3f} blur<={blur_thr:.0f} fft<={fft_thr:.3f}')
