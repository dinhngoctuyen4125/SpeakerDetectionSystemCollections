import kagglehub

DATA_ROOT = kagglehub.dataset_download("tranvannha/vivoice34")
NOISE_DIR = kagglehub.dataset_download("minhtu4n/musan-train-set")

print(DATA_ROOT)
print(NOISE_DIR)