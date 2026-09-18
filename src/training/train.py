import os
import shutil
from omegaconf import OmegaConf

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from clearml import Task

from src.models.lightning_module import GPTLightningModule
from src.data.wikitext_datamodule import WikiTextDataModule


def main():
    # ------------------------------------------------------------------
    # 1. Пути и настройки из переменных окружения
    # ------------------------------------------------------------------
    data_dir = os.environ.get("DATA_DIR", "/content/drive/MyDrive/CommonCrawl")
    checkpoint_dir = os.environ.get(
        "CHECKPOINT_DIR",
        "/content/drive/MyDrive/CommonCrawl/checkpoints",
    )
    project_name = os.environ.get("CLEARML_PROJECT_NAME", "MNNA-2026")

    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Чекпоинты будут сохраняться в: {checkpoint_dir}")

    # ------------------------------------------------------------------
    # 2. Конфиг
    # ------------------------------------------------------------------
    cfg = OmegaConf.load("/content/MNNA-2026/configs/gpt_config.yaml")
    print("Конфиг:")
    print(OmegaConf.to_yaml(cfg))

    # ------------------------------------------------------------------
    # 3. ClearML: инициализация ДО Trainer — так логи подхватятся автоматически
    # ------------------------------------------------------------------
    task = Task.init(
        project_name=project_name,
        task_name="GPT training WikiText-103",
        task_type=Task.TaskTypes.training,
        auto_connect_frameworks={
            "tensorboard": True,   # автоматически тянет TB-скаляры
            "pytorch": False,      # не оборачиваем модель, чтобы не мешать Lightning
            "matplotlib": False,
            "logger": False,
        },
        reuse_last_task_id=False,
    )
    task.connect(OmegaConf.to_container(cfg, resolve=True))
    print(f"ClearML Task ID: {task.id}")
    print(f"ClearML URL: {task.get_output_log_web_page()}")

    # ------------------------------------------------------------------
    # 4. DataModule
    # ------------------------------------------------------------------
    npz_path = os.path.join(data_dir, "wikitext_packed_batches.npz")
    print(f"Датасет: {npz_path}")
    assert os.path.exists(npz_path), f"Нет файла: {npz_path}"

    dm = WikiTextDataModule(
        npz_path=npz_path,
        batch_size=cfg.training.batch_size,
        val_split=0.05,
    )

    # ------------------------------------------------------------------
    # 5. Модель
    # ------------------------------------------------------------------
    model = GPTLightningModule(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Модель создана. Параметров: {n_params:,}")

    # ------------------------------------------------------------------
    # 6. Callbacks
    # ------------------------------------------------------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="gpt-{epoch:02d}-{val_perplexity:.2f}",
        monitor="val/perplexity",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=2,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ------------------------------------------------------------------
    # 7. TensorBoard logger
    # ------------------------------------------------------------------
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(data_dir, "logs"),
        name="gpt",
    )

    # ------------------------------------------------------------------
    # 8. Trainer
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        accelerator, devices, precision = "gpu", 1, "16-mixed"
        print("GPU доступен, используем 16-mixed")
    else:
        accelerator, devices, precision = "cpu", 1, "32-true"
        print("GPU недоступен — обучение пойдёт на CPU (медленно!)")

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=cfg.training.max_grad_norm,
        gradient_clip_algorithm="norm",
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,
        log_every_n_steps=50,
        val_check_interval=0.5,
        precision=precision,
    )

    # ------------------------------------------------------------------
    # 9. Resume
    # ------------------------------------------------------------------
    last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
    resume_path = last_ckpt if os.path.exists(last_ckpt) else None
    if resume_path:
        print(f"Продолжаем обучение с чекпоинта: {resume_path}")
    else:
        print("Чекпоинт не найден — обучение с нуля.")

    # ------------------------------------------------------------------
    # 10. Запуск
    # ------------------------------------------------------------------
    trainer.fit(model, datamodule=dm, ckpt_path=resume_path)

    # ------------------------------------------------------------------
    # 11. Финальные копии
    # ------------------------------------------------------------------
    final_path = os.path.join(checkpoint_dir, "final.ckpt")
    trainer.save_checkpoint(final_path)
    print(f"\nФинальный чекпоинт сохранён: {final_path}")

    if checkpoint_callback.best_model_path:
        best_copy = os.path.join(checkpoint_dir, "best.ckpt")
        shutil.copy(checkpoint_callback.best_model_path, best_copy)
        print(f"Лучший чекпоинт скопирован: {best_copy}")

    print("\nЛучший чекпоинт:", checkpoint_callback.best_model_path)
    print("Лучшая perplexity:", checkpoint_callback.best_model_score)

    # Закрываем задачу ClearML явно
    task.close()


if __name__ == "__main__":
    main()
