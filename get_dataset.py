import os
import kagglehub
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str)
args = parser.parse_args()

ROOT_DIR = os.path.join(os.getcwd(), args.datasets)

DATA_ROOT = kagglehub.dataset_download(
    "tranvannha/vivoice34",
    output_dir=os.path.join(ROOT_DIR, "vivoice34")
)

NOISE_DIR = kagglehub.dataset_download(
    "minhtu4n/musan-train-set",
    output_dir=os.path.join(ROOT_DIR, "musan")
)

print(os.path.join(DATA_ROOT, "VoxVietnamese_dataset"))
print(NOISE_DIR)