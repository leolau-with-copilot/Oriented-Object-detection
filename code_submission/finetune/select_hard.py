"""Flag training images that contain SMALL and/or DARK ships — the failure modes
our miss-analysis surfaced. Writes hard_images.txt (one image stem per line) used
to oversample these images during fine-tuning.
"""
import os, sys, glob, cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

IMG_DIR = paths.IMG_TRAIN
LBL_DIR = paths.LBL_TRAIN
HARD_TXT = os.path.join(paths.DATA, 'hard_images.txt')
SMALL_PX = 4000.0     # ship polygon area below this = "small"
DARK_INT = 85.0       # mean interior brightness below this (0-255) = "dark"

hard, n_small, n_dark = [], 0, 0
for ip in sorted(glob.glob(os.path.join(IMG_DIR, '*'))):
    stem = os.path.splitext(os.path.basename(ip))[0]
    lp = os.path.join(LBL_DIR, stem + '.txt')
    if not (os.path.exists(lp) and os.path.getsize(lp)):
        continue
    img = cv2.imread(ip)
    if img is None:
        continue
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    is_small = is_dark = False
    for line in open(lp):
        a = line.split()
        if len(a) < 9:
            continue
        poly = (np.array(a[1:9], np.float32).reshape(4, 2) * [W, H]).astype(np.int32)
        area = cv2.contourArea(poly.astype(np.float32))
        mask = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(mask, poly, 255)
        inten = float(gray[mask > 0].mean()) if (mask > 0).any() else 255.0
        if area < SMALL_PX:
            is_small = True
        if inten < DARK_INT:
            is_dark = True
    if is_small or is_dark:
        hard.append(stem)
    n_small += is_small
    n_dark += is_dark

with open(HARD_TXT, 'w') as f:
    f.write('\n'.join(hard) + ('\n' if hard else ''))

total = len(glob.glob(os.path.join(IMG_DIR, '*')))
print(f'train images        : {total}')
print(f'with SMALL ship     : {n_small}')
print(f'with DARK ship      : {n_dark}')
print(f'HARD (small or dark): {len(hard)}  -> hard_images.txt')
