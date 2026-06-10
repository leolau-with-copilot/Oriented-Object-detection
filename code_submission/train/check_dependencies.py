"""Dependency and dataset sanity check for the Alibaba Cloud training env.

Run after activating the conda env:
    conda activate ship
    python check_dependencies.py
"""
from pathlib import Path


def check_import(name, import_name=None):
    import importlib

    mod = importlib.import_module(import_name or name)
    version = getattr(mod, "__version__", "unknown")
    print(f"OK  {name:<22} {version}")
    return mod


def main():
    root = Path(__file__).resolve().parent

    print("== Python package imports ==")
    torch = check_import("torch")
    check_import("torchvision")
    ultralytics = check_import("ultralytics")
    check_import("polars")
    check_import("seaborn")
    check_import("cv2", "cv2")
    check_import("pandas")
    check_import("numpy")
    check_import("yaml", "yaml")
    check_import("matplotlib")
    check_import("scipy")
    check_import("psutil")
    check_import("py-cpuinfo", "cpuinfo")
    check_import("tqdm")
    check_import("PIL", "PIL")
    check_import("requests")
    check_import("ultralytics-thop", "thop")

    print("\n== CUDA ==")
    print("torch cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    else:
        raise SystemExit("ERROR: CUDA is not available in this environment")

    print("\n== Ultralytics model YAML ==")
    from ultralytics import YOLO

    YOLO("yolo11s-obb.yaml")
    print("OK  loaded yolo11s-obb.yaml architecture")

    print("\n== Dataset paths ==")
    required = [
        root / "data.yaml",
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
        root / "test_images",
    ]
    for path in required:
        if not path.exists():
            raise SystemExit(f"ERROR: missing {path}")
        print(f"OK  {path.relative_to(root)}")

    print("\nAll dependency checks passed.")


if __name__ == "__main__":
    main()
