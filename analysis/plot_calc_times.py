"""Histogram of dxtb wall times per structure (paper Fig. 6).

Reads ``dxtb_calc_time`` from the packed dataset written by ``orbq-datagen prepare``.

    python analysis/plot_calc_times.py --tar prepared/tar/dataset.tar --out calc_times.png
"""

import argparse
import io
import multiprocessing as mp
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def _calc_time(byte_data):
    try:
        data = torch.load(io.BytesIO(byte_data), map_location="cpu", weights_only=False)
        if "dxtb_calc_time" in data:
            return float(data["dxtb_calc_time"])
    except Exception:
        return None
    return None


def _members(tar_path):
    with tarfile.open(tar_path, "r|") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".pt"):
                f = tar.extractfile(member)
                if f is not None:
                    yield f.read()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tar", type=Path, required=True, help="Packed dataset (prepared/tar/dataset.tar)."
    )
    parser.add_argument("--out", type=Path, default=Path("calc_time_histogram.png"))
    parser.add_argument(
        "--cache", type=Path, default=None, help="Optional .npy cache of extracted times."
    )
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args(argv)

    if args.cache is not None and args.cache.exists():
        calc_times = np.load(args.cache)
    else:
        with mp.Pool(args.num_workers) as pool:
            calc_times = np.array(
                [
                    t
                    for t in tqdm(
                        pool.imap_unordered(_calc_time, _members(args.tar), chunksize=100)
                    )
                    if t is not None
                ]
            )
        if args.cache is not None:
            np.save(args.cache, calc_times)

    if len(calc_times) == 0:
        print("No calculation times found.")
        return
    mean_time, median_time = np.mean(calc_times), np.median(calc_times)
    print(f"{len(calc_times)} calculations. Mean: {mean_time:.3f} s, median: {median_time:.3f} s")

    plt.figure(figsize=(6, 6))
    plt.hist(
        calc_times,
        bins=50,
        alpha=0.75,
        color="cornflowerblue",
        edgecolor="black",
        label=f"Mean: {mean_time:.3f} s\nMedian: {median_time:.3f} s",
    )
    plt.xlabel("Calculation time (s)", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
