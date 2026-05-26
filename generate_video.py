#
# Stitch per-frame PNG dumps under <model_path>/test/<subdir>/<cam>/
# into MP4 videos with ffmpeg.
#
# Expected folder layout (produced by train.py with --log_images):
#
#   <model_path>/
#     test/
#       gt/cam00/0001.png …
#       renders/cam00/0001.png …
#       motion/cam00/0002.png …            # may start at 0002
#       motion_label/cam00/0001.png.png …  # double .png is intentional
#
import os
import re
import sys
from argparse import ArgumentParser
from pathlib import Path


SPLIT = "test"
SUBDIRS = ("gt", "renders", "motion", "motion_label")
DEFAULT_FRAMERATE = 30


def run(cmd):
    print(f"==== {cmd}")
    err = os.system(cmd)
    if err:
        print(f"FATAL: ffmpeg failed (exit {err})")
        sys.exit(err)


def detect_pattern(cam_dir: Path):
    """Return (ffmpeg_input_pattern, start_number) or (None, None) if empty."""
    files = sorted(p.name for p in cam_dir.iterdir() if p.is_file())
    if not files:
        return None, None

    # motion_label uses 0001.png.png; everything else uses 0001.png
    if files[0].endswith(".png.png"):
        pattern = "%04d.png.png"
        rx = re.compile(r"^(\d{4})\.png\.png$")
    else:
        pattern = "%04d.png"
        rx = re.compile(r"^(\d{4})\.png$")

    indices = [int(m.group(1)) for f in files if (m := rx.match(f))]
    if not indices:
        return None, None
    return pattern, min(indices)


def make_video(cam_dir: Path, out_path: Path, framerate: int):
    pattern, start = detect_pattern(cam_dir)
    if pattern is None:
        print(f"  [skip] {cam_dir} (no PNG frames)")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"ffmpeg -y -thread_queue_size 1024 "
        f"-framerate {framerate} -start_number {start} "
        f"-i '{cam_dir}/{pattern}' "
        f"-vb 20M -pix_fmt yuv420p '{out_path}'"
    )
    run(cmd)


def generate(model_paths, framerate=DEFAULT_FRAMERATE):
    for scene_dir in model_paths:
        scene_dir = Path(scene_dir)
        print(f"\n=== Scene: {scene_dir}")

        split_dir = scene_dir / SPLIT
        if not split_dir.is_dir():
            print(f"  [skip] missing {split_dir}")
            continue

        for subdir in SUBDIRS:
            sub = split_dir / subdir
            if not sub.is_dir():
                continue

            for cam_dir in sorted(p for p in sub.iterdir() if p.is_dir()):
                out = scene_dir / f"output_{SPLIT}_{subdir}_{cam_dir.name}.mp4"
                print(f"  {SPLIT}/{subdir}/{cam_dir.name} -> {out.name}")
                make_video(cam_dir, out, framerate)


if __name__ == "__main__":
    parser = ArgumentParser(description="Stitch rendered frame PNGs into MP4 videos.")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str)
    parser.add_argument("--framerate", "-r", type=int, default=DEFAULT_FRAMERATE)
    args = parser.parse_args()
    generate(args.model_paths, framerate=args.framerate)
