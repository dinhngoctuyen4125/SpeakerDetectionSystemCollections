import argparse
import torch
import os
from utils import VoxVietDataModule
from ecapa_tdnn import LitSpeakerModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

parser = argparse.ArgumentParser()

parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--noise_dir", type=str, required=True)

args = parser.parse_args()

DATA_ROOT = os.path.join(args.data_root, "VoxVietnamese_dataset")
NOISE_DIR = os.path.join(args.noise_dir, "musan")

CHECKPOINT_DIR = "checkpoints"

dm = VoxVietDataModule(
    train_dir   = os.path.join(DATA_ROOT, "train"),
    valid_dir   = os.path.join(DATA_ROOT, "val"), # Thư mục chứa folder wav
    trials_path = os.path.join(DATA_ROOT, "val", "trials.txt"), # File txt đánh giá
    batch_size  = 32,
    num_workers = 4,
    aug_ratio   = 0.5,           
    noise_dir   = NOISE_DIR,   
    use_augmentation=True
)

dm.setup()

# 2. Init Lightning Model
model = LitSpeakerModel(
    num_classes=dm.num_classes,
    learning_rate_backbone=1e-5,
    learning_rate_head=1e-3
)

# 3. Init Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath   ="checkpoints",
    filename  ="best_model", 
    monitor   ="val_eer",      
    mode      ="min",              
    save_top_k=1,          
    verbose   =True
)
early_stop_callback = EarlyStopping(
    monitor ="val_eer",
    mode    ="min",         
    patience=5,
    verbose =True
)

# 4. Trainer
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1, 
    callbacks=[checkpoint_callback, early_stop_callback],
    enable_progress_bar=True,
    num_sanity_val_steps=0,
    precision="16-mixed",
    accumulate_grad_batches=4
)

# 5. Train
trainer.fit(model, datamodule=dm)

# 6. TRÍCH XUẤT TRỌNG SỐ ĐÃ FineTuned
best_ckpt_path = "checkpoints/best_model.ckpt"
print(f"Best model saved at: {best_ckpt_path}")

lightning_checkpoint = torch.load(best_ckpt_path, map_location="cpu")
state_dict = lightning_checkpoint['state_dict']

fine_tuned_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('backbone.'):
        fine_tuned_state_dict[k[9:]] = v