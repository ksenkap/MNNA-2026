import os
from dotenv import load_dotenv
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.lightning_module import GPTLightningModule
from src.data.wikitext_datamodule import WikiTextDataModule


def main():
    # ---------------------------------------------------------------
    # 1. Переменные окружения
    # ---------------------------------------------------------------
    # .env может не быть — это не критично для ЛР4
    load_dotenv("/content/MNNA-2026/.env", override=False)

    project_name = os.getenv("CLEARML_PROJECT_NAME", "MNNA-2026")

    # ---------------------------------------------------------------
    # 2. Конфиг
    # ---------------------------------------------------------------
    cfg = OmegaConf.load("/content/MNNA-2026/configs/gpt_config.yaml")
    print("Конфиг:")
    print(OmegaConf.to_yaml(cfg))

    # Пути берём из конфига
    checkpoint_dir = cfg.training.checkpoint_dir
    log_dir = cfg.training.log_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Чекпоинты: {checkpoint_dir}")
    print(f"Логи:      {log_dir}")

    # ---------------------------------------------------------------
    # 3. DataModule — используем НОВЫЙ датасет из ЛР4
    # ---------------------------------------------------------------
    data_dir = "/content/drive/MyDrive/CommonCrawl"
    npz_path = os.path.join(data_dir, "wikitext_packed_batches_large.npz")
    assert os.path.exists(npz_path), f"Нет датасета: {npz_path}"

    dm = WikiTextDataModule(
        npz_path=npz_path,
        batch_size=cfg.training.batch_size,
        val_split=0.05,
    )

    # ---------------------------------------------------------------
    # 4. Модель (GPT + GQA внутри attention)
    # ---------------------------------------------------------------
    model = GPTLightningModule(cfg)

    # ---------------------------------------------------------------
    # 5. Callbacks
    # ---------------------------------------------------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="gpt-{epoch:02d}-{val_perplexity:.2f}",
        monitor="val/perplexity",
        mode="min",
        save_top_k=3,
        save_last=True,           # сохраняем last.ckpt для resume
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ---------------------------------------------------------------
    # 6. Логгер (TensorBoard)
    # ---------------------------------------------------------------
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name="gpt",
    )

    # ---------------------------------------------------------------
    # 7. Опционально: ClearML — если есть .env с ключами
    # ---------------------------------------------------------------
    clearml_logger = None
    try:
        from clearml import Task
        access_key = os.getenv("CLEARML_API_ACCESS_KEY")
        secret_key = os.getenv("CLEARML_API_SECRET_KEY")
        if access_key and secret_key:
            task = Task.init(
                project_name=project_name,
                task_name="Lab4 training",
                auto_connect_frameworks=False,
            )
            print(f"ClearML подключён, task id: {task.id}")
        else:
            print("ClearML не подключён (нет ключей в .env) — это ок")
    except Exception as e:
        print(f"ClearML недоступен: {e}")

    # ---------------------------------------------------------------
    # 8. Trainer
    # ---------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=cfg.training.max_grad_norm,
        gradient_clip_algorithm="norm",
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,
        log_every_n_steps=50,
        val_check_interval=0.5,     # проверка валидации дважды за эпоху
        precision="16-mixed",       # fp16 на T4
    )

    # ---------------------------------------------------------------
    # 9. Resume из last.ckpt, если есть
    # ---------------------------------------------------------------
    last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
    resume_path = last_ckpt if os.path.exists(last_ckpt) else None
    if resume_path:
        print(f"Продолжаем обучение с: {resume_path}")
    else:
        print("Начинаем обучение с нуля.")

    # ---------------------------------------------------------------
    # 10. Запуск
    # ---------------------------------------------------------------
    trainer.fit(model, datamodule=dm, ckpt_path=resume_path)

    print("\n=== Обучение завершено ===")
    print("Лучший чекпоинт:", checkpoint_callback.best_model_path)
    print("Лучшая perplexity:", checkpoint_callback.best_model_score)


if __name__ == "__main__":
    main()
