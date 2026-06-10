"""Fine-tune the from-scratch YOLO11s-OBB checkpoint (best.pt) with extra focus on
SMALL and DARK ships — the failure modes the miss-analysis surfaced.

Pipeline (all paths relative to this script's folder, so it is location-independent):
  1. read hard_images.txt (images containing small/dark ships, from select_hard.py)
  2. OVERSAMPLE them by symlinking distinct copies into images/hard_os + labels/hard_os
     (distinct filenames => Ultralytics treats them as extra samples, not duplicates)
  3. write train_list.txt (all originals once + the oversampled hard images) and data_finetune.yaml
  4. load best.pt and CONTINUE training with a gentle, small-object-friendly recipe.

Recipe (the configuration that actually produced best_finetuned.pt):
    imgsz=1024            # SAME resolution as the base training (do NOT upscale)
    freeze=10             # freeze the backbone, only adapt the head
    lr0=1e-3, no warmup   # gentle — don't wipe what the model already learned
    mosaic=0              # keep small targets intact, no synthetic context
    load best.pt directly # NOTE: never pass pretrained=False — it reinitialises the
                          #       loaded weights and silently restarts FROM SCRATCH.

Competition compliance: best.pt was trained from scratch on the PROVIDED data only
(no external data, no external pretraining). This step merely continues training that
same checkpoint on the same data — it introduces no outside prior knowledge.
"""
import os, sys, glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

ROOT = paths.DATA                                     # working dir: data root
IMG_TRAIN = paths.IMG_TRAIN
LBL_TRAIN = paths.LBL_TRAIN
IMG_OS = os.path.join(paths.DATA, 'images', 'hard_os')
LBL_OS = os.path.join(paths.DATA, 'labels', 'hard_os')
HARD_TXT = os.path.join(paths.DATA, 'hard_images.txt')
OVERSAMPLE = 3          # hard images appear 3x total (1 original + 2 extra symlinks)
EXTS = ('.jpg', '.jpeg', '.png')


def find(dir_, stem, exts=EXTS):
    for e in exts:
        p = os.path.join(dir_, stem + e)
        if os.path.exists(p):
            return p
    return None


def build_oversampled_list():
    os.makedirs(IMG_OS, exist_ok=True)
    os.makedirs(LBL_OS, exist_ok=True)
    hard = [l.strip() for l in open(HARD_TXT) if l.strip()]

    entries = [p for p in sorted(glob.glob(os.path.join(IMG_TRAIN, '*')))
               if p.lower().endswith(EXTS)]
    n_extra = 0
    for stem in hard:
        ip = find(IMG_TRAIN, stem)
        lp = find(LBL_TRAIN, stem, ('.txt',))
        if not (ip and lp):
            continue
        ext = os.path.splitext(ip)[1]
        for k in range(1, OVERSAMPLE):
            li = os.path.join(IMG_OS, f'{stem}__os{k}{ext}')
            ll = os.path.join(LBL_OS, f'{stem}__os{k}.txt')
            for src, dst in [(ip, li), (lp, ll)]:
                if os.path.islink(dst) or os.path.exists(dst):
                    os.remove(dst)
                os.symlink(src, dst)
            entries.append(li)
            n_extra += 1

    list_path = os.path.join(ROOT, 'train_list.txt')
    with open(list_path, 'w') as f:
        f.write('\n'.join(entries) + '\n')
    print(f'train list: {len(entries)} entries '
          f'({len(entries) - n_extra} originals + {n_extra} oversampled hard)')
    return list_path


def write_yaml(train_list):
    import yaml
    yp = os.path.join(ROOT, 'data_finetune.yaml')
    yaml.safe_dump({'path': ROOT, 'train': train_list,
                    'val': os.path.join(ROOT, 'images', 'val'),
                    'names': {0: 'ship'}}, open(yp, 'w'), sort_keys=False)
    return yp


def main():
    from ultralytics import YOLO
    train_list = build_oversampled_list()
    data_yaml = write_yaml(train_list)

    model = YOLO(paths.BEST)                          # continue from our from-scratch weights

    model.train(
        data=data_yaml,
        epochs=40, patience=20,
        imgsz=1024, batch=16,                        # SAME resolution as base training
        freeze=10,                                   # freeze backbone, adapt the head only
        lr0=1e-3, lrf=0.01, warmup_epochs=0.0,       # gentle fine-tune, no warmup
        degrees=180.0,                               # keep full-angle OBB coverage
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.5,           # stronger value jitter -> dark-ship robustness
        scale=0.3,                                   # less down-scaling -> small targets stay resolvable
        mosaic=0.0,                                  # keep small targets intact
        fliplr=0.5, flipud=0.5,
        mixup=0.0, copy_paste=0.0,
        # IMPORTANT: do NOT set pretrained=False here — it would reinitialise best.pt.
        project=os.path.join(paths.DATA, 'runs_finetune'), name='small_dark_1024', exist_ok=True,
    )
    print('\nDONE. fine-tuned weights at <data>/runs_finetune/small_dark_1024/weights/best.pt')
    print('Then run TTA inference:  python ../inference/tta_submission.py')


if __name__ == '__main__':
    main()
