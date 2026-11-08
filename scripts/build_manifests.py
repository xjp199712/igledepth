"""Build train / test manifests for the revision runs (CPU only).

Train: Eigen train split, frames that have clear RGB + sparse LiDAR on disk.
Each frame lists every weather rendering that exists on disk. Weather domains that are
missing (fog 75m/150m, mix_rain 50mm until they are uploaded) are simply absent and
reported. Re-run after adding data; it is idempotent.

Test: reuses the author-697 manifests (sunny, raingan, snowgan, mix_snow) and the real
weather manifests, and derives fog / mix_rain test manifests when those files exist.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import REPO, ROOT, save_json  # noqa: E402

DATA = f"{ROOT}/data"
OUT = f"{ROOT}/revision/manifests"
WEATHER_ROOT = f"{DATA}/weather-full"
# (domain name, folder, sub-folder replacing "data")
DOMAINS = [
    ("raingan", "raingan", "data"),
    ("snowgan", "snowgan", "data"),
    ("mix_snow", "mix_snow", "data"),
    ("mix_rain", "mix_rain", "50mm"),
    ("fog75", "fog", "75m"),
    ("fog150", "fog", "150m"),
]


def weather_path(rel, folder, sub):
    # rel: 2011_09_26/2011_09_26_drive_0057_sync/image_02/data/0000000116.png
    drive_rel = rel.split("/", 1)[1]
    return os.path.join(WEATHER_ROOT, folder, drive_rel.replace("/data/", f"/{sub}/"))


def build_train():
    split = f"{REPO}/data_split/kitti_depth/eigen_train_files_with_gt.txt"
    frames, counts = [], {d[0]: 0 for d in DOMAINS}
    skipped = 0
    for line in open(split):
        parts = line.split()
        if len(parts) < 2 or parts[1] == "None":
            continue
        rel, dep = parts[0], parts[1]
        rgb = f"{DATA}/kitti-full-paired/rgb/{rel}"
        lidar = f"{DATA}/kitti-full-paired/depth/{dep}"
        if not (os.path.exists(rgb) and os.path.exists(lidar)):
            skipped += 1
            continue
        weather = {}
        for name, folder, sub in DOMAINS:
            p = weather_path(rel, folder, sub)
            if os.path.exists(p):
                weather[name] = p
                counts[name] += 1
        frames.append({
            "id": rel,
            "rgb": rgb,
            "lidar": lidar,
            "pseudo": f"{ROOT}/revision/pseudo_dav2/{rel}",
            "weather": weather,
        })
    save_json({"frames": frames, "weather_counts": counts, "skipped_missing_rgb_or_lidar": skipped},
              f"{OUT}/train.json")
    print(f"train frames {len(frames)} (skipped {skipped}); weather coverage {counts}")


def build_test():
    src = f"{DATA}/experiment-preparation-20261007/author-697"
    tests = {}
    sunny = json.load(open(f"{src}/test-sunny.json"))["samples"]
    for dom in ["sunny", "raingan", "snowgan", "mix_snow"]:
        tests[f"kitti_{dom}"] = json.load(open(f"{src}/test-{dom}.json"))["samples"]
    for name, folder, sub in DOMAINS:
        if name in ("raingan", "snowgan", "mix_snow"):
            continue
        samples = []
        for s in sunny:
            p = weather_path(s["scene_id"], folder, sub)
            if os.path.exists(p):
                samples.append(dict(s, domain=name, rgb_path=p, rgb_hw=None))
        if len(samples) == len(sunny):
            tests[f"kitti_{name}"] = samples
        print(f"test kitti_{name}: {len(samples)}/{len(sunny)} files present")
    real = f"{DATA}/experiment-preparation-20261007/manifests"
    tests["ds_rain"] = json.load(open(f"{real}/test-drivingstereo-rainy.json"))["samples"]
    tests["ds_fog"] = json.load(open(f"{real}/test-drivingstereo-foggy.json"))["samples"]
    tests["cadc_snow"] = json.load(open(f"{real}/test-cadc-real_snow.json"))["samples"]
    for k, v in tests.items():
        save_json({"name": k, "samples": v}, f"{OUT}/test_{k}.json")
    print({k: len(v) for k, v in tests.items()})


if __name__ == "__main__":
    build_train()
    build_test()
