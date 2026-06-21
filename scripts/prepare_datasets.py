#!/usr/bin/env python3
"""Download MNIST and CIFAR-10 into examples/datasets/RawData."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import urllib.request

import torchvision
import torchvision.transforms as transforms

# Domestic mirrors (faster inside mainland China than official overseas hosts).
MNIST_RAW_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]
MNIST_MIRROR_BASES = [
    "https://mirrors.aliyun.com/pytorch-models/mnist/MNIST/raw",
    "https://ossci-datasets.torchvision.org/mnist",
]
CN_MIRRORS = {
    "cifar10": [
        "https://mirrors.bfsu.edu.cn/osdn//datasets/74526/cifar-10-python.tar.gz",
        "https://mirrors.aliyun.com/pytorch-models/cifar-10-python.tar.gz",
    ],
}

CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR10_DIR = "cifar-10-batches-py"
CIFAR10_SIZE = 170498071
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"


def _md5_file(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_with_resume(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    resume_pos = os.path.getsize(dest) if os.path.exists(dest) else 0
    headers = {}
    if resume_pos > 0:
        headers["Range"] = f"bytes={resume_pos}-"
        print(f"[prepare] resume {dest} from byte {resume_pos}")

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        mode = "ab" if resume_pos > 0 and resp.status == 206 else "wb"
        if mode == "wb":
            resume_pos = 0
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) + resume_pos if total else None
        downloaded = resume_pos
        chunk_size = 1024 * 1024
        with open(dest, mode) as out:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = downloaded * 100.0 / total_bytes
                    print(
                        f"\r[prepare] {os.path.basename(dest)} "
                        f"{downloaded}/{total_bytes} ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
        print()


def _try_mirror_download(name: str, dest: str, mirrors: list[str]) -> bool:
    for url in mirrors:
        try:
            print(f"[prepare] try mirror: {url}")
            _download_with_resume(url, dest)
            return True
        except Exception as exc:
            print(f"[prepare] mirror failed: {exc}")
    return False


def prepare_cifar10_cn(root: str) -> None:
    archive_path = os.path.join(root, CIFAR10_ARCHIVE)
    extract_dir = os.path.join(root, CIFAR10_DIR)

    if os.path.isdir(extract_dir):
        print(f"[prepare] CIFAR-10 already extracted at {extract_dir}")
        return

    need_download = True
    if os.path.exists(archive_path):
        size = os.path.getsize(archive_path)
        if size >= CIFAR10_SIZE - 1024:
            md5 = _md5_file(archive_path)
            if md5 == CIFAR10_MD5:
                print(f"[prepare] reuse existing archive {archive_path}")
                need_download = False
            else:
                print(f"[prepare] archive md5 mismatch ({md5}), re-download")
        else:
            print(f"[prepare] partial archive ({size} bytes), resume download")

    if need_download:
        print("[prepare] downloading CIFAR-10 from domestic mirror ...")
        ok = _try_mirror_download("cifar10", archive_path, CN_MIRRORS["cifar10"])
        if not ok:
            raise RuntimeError("all domestic CIFAR-10 mirrors failed")

        md5 = _md5_file(archive_path)
        if md5 != CIFAR10_MD5:
            raise RuntimeError(
                f"CIFAR-10 md5 mismatch: got {md5}, expected {CIFAR10_MD5}"
            )

    print(f"[prepare] extracting {archive_path} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=root)

    if not os.path.isdir(extract_dir):
        raise RuntimeError(f"extract failed, missing {extract_dir}")

    # Verify torchvision can load without hitting the official host again.
    torchvision.datasets.CIFAR10(
        root, download=False, train=True, transform=transforms.ToTensor()
    )
    torchvision.datasets.CIFAR10(
        root, download=False, train=False, transform=transforms.ToTensor()
    )
    print("[prepare] CIFAR-10 ready")


def prepare_cifar10_official(root: str) -> None:
    print("[prepare] downloading CIFAR-10 from official torchvision source ...")
    torchvision.datasets.CIFAR10(
        root, download=True, train=True, transform=transforms.ToTensor()
    )
    torchvision.datasets.CIFAR10(
        root, download=False, train=False, transform=transforms.ToTensor()
    )
    print("[prepare] CIFAR-10 ready")


def prepare_mnist_cn(root: str) -> None:
    raw_dir = os.path.join(root, "MNIST", "raw")
    if all(os.path.isfile(os.path.join(raw_dir, name)) for name in MNIST_RAW_FILES):
        print(f"[prepare] MNIST already exists at {raw_dir}")
    else:
        os.makedirs(raw_dir, exist_ok=True)
        print("[prepare] downloading MNIST from domestic mirror ...")
        for filename in MNIST_RAW_FILES:
            dest = os.path.join(raw_dir, filename)
            if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                print(f"[prepare] reuse {filename}")
                continue
            mirrors = [f"{base}/{filename}" for base in MNIST_MIRROR_BASES]
            if not _try_mirror_download("mnist", dest, mirrors):
                raise RuntimeError(f"all domestic MNIST mirrors failed for {filename}")

    torchvision.datasets.MNIST(
        root, download=False, train=True, transform=transforms.ToTensor()
    )
    torchvision.datasets.MNIST(
        root, download=False, train=False, transform=transforms.ToTensor()
    )
    print("[prepare] MNIST ready")


def prepare_mnist_official(root: str) -> None:
    mnist_dir = os.path.join(root, "MNIST")
    if os.path.isdir(mnist_dir):
        print(f"[prepare] MNIST already exists at {mnist_dir}")
        return

    print("[prepare] downloading MNIST ...")
    torchvision.datasets.MNIST(
        root, download=True, train=True, transform=transforms.ToTensor()
    )
    torchvision.datasets.MNIST(
        root, download=False, train=False, transform=transforms.ToTensor()
    )
    print("[prepare] MNIST ready")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset root. Default: <fedcompass>/examples/datasets/RawData",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mnist", "cifar10"],
        choices=["mnist", "cifar10", "all"],
    )
    parser.add_argument(
        "--mirror",
        choices=["official", "cn"],
        default="cn",
        help="Download source. 'cn' uses domestic mirrors for MNIST and CIFAR-10.",
    )
    args = parser.parse_args()

    root = os.path.abspath(
        args.data_dir
        or os.path.join(
            os.path.dirname(__file__), "..", "examples", "datasets", "RawData"
        )
    )
    os.makedirs(root, exist_ok=True)
    selected = set(args.datasets)
    if "all" in selected:
        selected = {"mnist", "cifar10"}

    print(f"[prepare] data_dir={root}")
    print(f"[prepare] mirror={args.mirror}")

    if "mnist" in selected:
        if args.mirror == "cn":
            prepare_mnist_cn(root)
        else:
            prepare_mnist_official(root)

    if "cifar10" in selected:
        if args.mirror == "cn":
            prepare_cifar10_cn(root)
        else:
            prepare_cifar10_official(root)

    print("[prepare] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
