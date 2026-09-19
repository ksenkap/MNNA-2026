import math
import torch
import pytorch_lightning as pl

from src.models.gpt import GPT


class GPTLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cfg"])

        n_kv_heads = cfg.model.get("n_kv_heads", None)

        self.model = GPT(
            vocab_size=cfg.model.vocab_size,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_layers=cfg.model.n_layers,
            d_ff=cfg.model.d_ff,
            n_kv_heads=n_kv_heads,
            max_seq_len=cfg.model.max_seq_len,
            dropout=cfg.model.dropout,
        )

    def forward(self, input_ids, seq_ids):
        logits, _ = self.model(input_ids, seq_ids, use_cache=False)
        return logits

    def _shared_step(self, batch, stage: str):
        input_ids, seq_ids = batch
        logits = self(input_ids, seq_ids)
        loss = self.model.compute_loss(logits, input_ids, seq_ids)
        perplexity = torch.exp(loss)

        self.log(f"{stage}/loss", loss, prog_bar=True,
                 on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/perplexity", perplexity, prog_bar=True,
                 on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )

        warmup_steps = self.cfg.training.warmup_steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = (current_step - warmup_steps) / max(1, 200_000)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr",
            },
        }

    def on_after_backward(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.log("train/grad_norm", total_norm, on_step=True, on_epoch=False)
