import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve

from wespeaker.models.projections import ArcMarginProduct as ArcMarginProjection
from speechbrain.inference.speaker import EncoderClassifier

from utils import compute_fbank

class WrappedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )
        self.model = classifier.mods.embedding_model

    def forward(self, feats):
        wav_lens = torch.ones(feats.shape[0], device=feats.device)
        emb = self.model(feats, wav_lens) # (B, 1, 192)
        return emb.squeeze(1) # (B, 192)
    
class LitSpeakerModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate_backbone=1e-5, learning_rate_head=1e-3, freeze_backbone_epochs=3):
        super().__init__()
        self.save_hyperparameters()
        self.best_val_eer = 1.0
        self.backbone = WrappedBackbone()
        embed_dim = 192
        
        self.projection = ArcMarginProjection(in_features=embed_dim, out_features=num_classes, margin=0.5, scale=64.0)
        self.criterion = nn.CrossEntropyLoss()
        self.validation_step_outputs = []

    def forward(self, x):
        out = self.backbone(x)
        if isinstance(out, tuple): out = out[-1]
        return out

    def extract_representative_embedding(self, wav):
        # Logic: 3s windows, 50% overlap
        window_size = 16000 * 3
        stride = window_size // 2
        
        if wav.size(1) <= window_size:
            segments = [F.pad(wav, (0, window_size - wav.size(1)))]
        else:
            segments = []
            for i in range(0, wav.size(1) - window_size + 1, stride):
                segments.append(wav[:, i:i+window_size])
            if (wav.size(1) - window_size) % stride != 0:
                segments.append(wav[:, -window_size:])
        
        segment_embeddings = []
        for seg in segments:
            fbank = compute_fbank(seg.to(self.device))
            emb = self.forward(fbank.unsqueeze(0))
            segment_embeddings.append(F.normalize(emb, p=2, dim=1))
        
        # Average -> Normalize
        avg_emb = torch.mean(torch.stack(segment_embeddings), dim=0)
        return F.normalize(avg_emb, p=2, dim=1)

    def training_step(self, batch, batch_idx):
        feats, labels = batch
        embeds = self.forward(feats)
        logits = self.projection(embeds, labels)
        loss = self.criterion(logits, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        for wav1, wav2, label in batch:
            emb1 = self.extract_representative_embedding(wav1)
            emb2 = self.extract_representative_embedding(wav2)
            score = torch.sum(emb1 * emb2).item()
            self.validation_step_outputs.append({"score": score, "label": label.item()})

    def on_validation_epoch_end(self):
        scores = np.array([x["score"] for x in self.validation_step_outputs])
        labels = np.array([x["label"] for x in self.validation_step_outputs])
        self.validation_step_outputs.clear()

        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
        
        # minDCF (p_target=0.01)
        min_dcf = np.min(fnr + 99 * fpr) / 100
        
        self.log('val_eer', eer, prog_bar=True)
        print(f"\n--- Epoch {self.current_epoch} Evaluation ---")
        print(f"EER: {eer:.4f} | minDCF: {min_dcf:.4f}")
        
        if eer < self.best_val_eer:
            self.best_val_eer = eer
            print(f"NEW BEST EER: {eer:.4f} ✨")

    def configure_optimizers(self):
        # Discriminative Learning Rates
        optimizer = torch.optim.Adam([
            {"params": self.backbone.parameters(), "lr": self.hparams.learning_rate_backbone},
            {"params": self.projection.parameters(), "lr": self.hparams.learning_rate_head},
        ], weight_decay=1e-4)
        
        # Lấy tổng số batch trong 1 epoch (an toàn trong PyTorch Lightning)
        # Nếu dùng nhiều GPU (DDP), số này sẽ tự động chia đều.
        total_steps = self.trainer.estimated_stepping_batches
        
        # Số steps trong 1 epoch
        steps_per_epoch = total_steps // self.trainer.max_epochs
        
        # T_0 = 10 epochs (quy đổi ra số steps)
        t0_steps = 10 * steps_per_epoch
        
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=t0_steps,
                T_mult=1,
                eta_min=1e-6
            ),
            "interval": "step", # Cập nhật mượt mà sau mỗi batch
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
    def on_train_epoch_start(self):
        if self.current_epoch < self.hparams.freeze_backbone_epochs:
            for p in self.backbone.parameters(): p.requires_grad = False
        else:
            for p in self.backbone.parameters(): p.requires_grad = True