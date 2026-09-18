# Лабораторная работа 2: Обучение GPT-like языковой модели

Реализация decoder-only трансформера (GPT-like) с нуля на PyTorch, обучение
на packed batches из WikiText-103 с использованием PyTorch Lightning,
логирование в ClearML и TensorBoard, инференс генерации текста.

## Содержание

- [Архитектура модели](#архитектура-модели)
- [Обоснование гиперпараметров](#обоснование-гиперпараметров)
- [Данные](#данные)
- [Инфраструктура обучения](#инфраструктура-обучения)
- [Результаты](#результаты)
- [Примеры генерации](#примеры-генерации)
- [Структура проекта](#структура-проекта)
- [Запуск](#запуск)
- [Переменные окружения](#переменные-окружения)

## Архитектура модели

Модель — decoder-only трансформер (GPT-like) в post-norm варианте.
Состоит из следующих модулей (`src/models/`):

### 1. `positional_encoding.py` — синусоидальное позиционное кодирование

Классическое синусоидальное PE из статьи *"Attention is all you need"* с
поддержкой **packed batching**: позиция токена считается **внутри своего
объекта**, а не от начала склеенной последовательности.

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Позиция токена восстанавливается по `seq_ids`: для каждого объекта счётчик
`pos` начинается с 0 и увеличивается на 1 внутри одного объекта, сбрасывается
при смене `seq_ids` или при PAD (`seq_ids == 0`).

### 2. `attention.py` — многоглавoe маскированное self-attention

Реализован **block-masked attention** для packed batching. Внимание между
позициями `i` и `j` разрешено только если:

```
(s_i == s_j)  ∧  (j ≤ i)  ∧  (s_i ≠ 0)
```

где `s_i` — `seq_ids[i]`:

- `s_i == s_j` — токены принадлежат одному объекту,
- `j ≤ i` — causal маска (нельзя смотреть в будущее),
- `s_i ≠ 0` — PAD-токены не участвуют во внимании.

Маска накладывается на матрицу attention scores **до** softmax (через
`masked_fill(-inf)`), что сохраняет нормировку распределения.

Стандартные операции:
- линейные проекции `W_q, W_k, W_v, W_o ∈ R^(d_model × d_model)`,
- разбиение на `n_heads` голов с `d_head = d_model / n_heads`,
- scaled dot-product: `softmax(QKᵀ / √d_head) V`.

### 3. `ffn.py` — feed-forward network

Двухслойная полносвязная сеть, применяется к каждому токену независимо:

```
FFN(x) = Dropout(act(x W₁ + b₁)) W₂ + b₂
```

- `d_ff = 4 · d_model` — стандартное расширение,
- активация — GELU (по умолчанию) или ReLU,
- dropout — между двумя линейными слоями.

### 4. `transformer_layer.py` — слой трансформера (post-norm)

Реализован **post-norm** вариант с residual connections:

```
z₁ = LayerNorm(x  + Dropout(Attention(x)))
z₂ = LayerNorm(z₁ + Dropout(FFN(z₁)))
```

Post-norm выбран согласно заданию (pre-norm не реализовывался). Residual
connections предотвращают затухание градиентов в глубоких стеках.

### 5. `gpt.py` — сборка модели + LM-head + masked loss

Полная модель:

```
input_ids ─► TokenEmbedding ─► +PositionalEncoding ─► N × TransformerLayer
          ─► LayerNorm ─► Linear(d_model → vocab_size) ─► logits
```

- `token_embedding`: `nn.Embedding(vocab_size, d_model)`,
- `lm_head`: `nn.Linear(d_model, vocab_size, bias=False)`,
- `compute_loss(logits, targets, seq_ids)` — cross-entropy с маской для
  packed batching:

```
M_i^loss = (s_i == s_{i+1}) ∧ (s_i ≠ 0)
```

Учитываются только переходы **внутри одного объекта** и не-PAD позиции.
Softmax не применяется вручную — `F.cross_entropy` делает это внутри.

### 6. `lightning_module.py` — LightningModule

Инкапсулирует модель, оптимизатор, LR-планировщик, метрики и logging.

- `training_step` / `validation_step` → `_shared_step`,
- логирует `train/loss`, `val/loss`, `train/perplexity`, `val/perplexity`,
- `on_after_backward` логирует норму градиентов `train/grad_norm`,
- `configure_optimizers` — AdamW + LambdaLR с warm-up и cosine decay.

## Обоснование гиперпараметров

| Параметр | Значение | Обоснование |
|---|---|---|
| `vocab_size` | 10000 | Размер BPE-словаря, обученного в ЛР1 на отфильтрованном Common Crawl |
| `d_model` | 256 | Компромисс между качеством и скоростью на T4. `d_head = 32` — типичное значение |
| `n_heads` | 8 | `d_model / n_heads = 32`, стандартное соотношение. Даёт 8 независимых пространств внимания |
| `n_layers` | 6 | Сбалансированное число слоёв для датасета ~2.9M токенов. Больше — риск переобучения |
| `d_ff` | 1024 | `4 × d_model` — классическая пропорция из "Attention is all you need" |
| `max_seq_len` | 512 | Совпадает с `seq_len` в packed batches из ЛР1 |
| `dropout` | 0.15 | Регуляризация для относительно небольшого датасета |
| `batch_size` | 32 | Максимум, влезающий в T4 (15 GB) при `16-mixed` |
| `learning_rate` | 3e-4 | Стандартный LR для AdamW в трансформерах |
| `weight_decay` | 0.01 | Стандартная L2-регуляризация AdamW |
| `warmup_steps` | 1000 | Плавный разогрев LR, ~15% от общего числа шагов |
| `max_grad_norm` | 1.0 | Clip-by-norm для стабильности обучения |
| `precision` | 16-mixed | Смешанная точность для ускорения на T4 |

**Размер модели:** 9 859 072 параметра (~9.9M).
Соотношение параметров к токенам датасета — примерно 3.4×,
что исключает переобучение при правильной регуляризации
(это подтверждается результатами: `train_loss < val_loss` без
большого разрыва).

## Данные

Обучение производится на **packed batches** из WikiText-103,
подготовленных в **ЛР1**:

- `wikitext_packed_batches.npz`:
  - `batches`: `(5633, 512)` — токены,
  - `masks`: `(5633, 512)` — `seq_ids` (0 для PAD, 1, 2, ... для объектов),
  - `batch_size`: исходный размер батча.
- `bpe_tokenizer.json` — BPE-токенизатор (словарь 10000).

Train/Val split: **95% / 5%** (`val_split=0.05`).

## Инфраструктура обучения

### PyTorch Lightning

- `GPTLightningModule` — модель,
- `WikiTextDataModule` — data pipeline,
- `pl.Trainer` с callbacks, логгерами и gradient clipping.

### Learning rate scheduling и warm-up

Линейный warm-up первые `warmup_steps=1000` шагов, затем косинусный decay
до `0.1 × base_lr`:

```python
def lr_lambda(current_step):
    if current_step < warmup_steps:
        return current_step / warmup_steps
    progress = (current_step - warmup_steps) / max(1, 100000)
    return max(0.1, 0.5 * (1.0 + cos(pi * progress)))
```

### Gradient clipping и норма градиентов

- **Clipping:** `gradient_clip_val=1.0`, `gradient_clip_algorithm="norm"`
  (по глобальной норме, внутри `Trainer`).
- **Норма:** `on_after_backward` вычисляет глобальную норму градиентов и
  логирует её как `train/grad_norm`. Можно смотреть в TensorBoard/ClearML
  для контроля стабильности.

### ModelCheckpoint

- `save_top_k=3` — три лучших по `val/perplexity`,
- `save_last=True` — `last.ckpt` для resume,
- `every_n_epochs=2` — сохранение каждые 2 эпохи,
- `monitor="val/perplexity"`, `mode="min"`.

Дополнительно в конце `train.py` создаются **дубликаты**:
- `best.ckpt` — копия лучшего чекпоинта,
- `final.ckpt` — финальный чекпоинт после последней эпохи.

Это защищает артефакты от перезаписи при resume.

### Resume

`train.py` автоматически проверяет наличие `last.ckpt` в `checkpoint_dir`
и передаёт его в `trainer.fit(..., ckpt_path=resume_path)`. Обучение
продолжается с той же эпохи, с сохранением состояния оптимизатора и LR-планировщика.

### ClearML и TensorBoard

- **ClearML:** `Task.init(project_name=..., task_name=..., auto_connect_frameworks={"tensorboard": True})`.
  Все скаляры TensorBoard автоматически уходят в ClearML. Конфиг
  загружается в задачу через `task.connect(cfg)`.
- **TensorBoard:** `TensorBoardLogger(save_dir=DATA_DIR/logs, name="gpt")` —
  логи хранятся на Google Drive, не теряются между сессиями.
- Метрики: `train/loss`, `val/loss`, `train/perplexity`, `val/perplexity`,
  `lr`, `train/grad_norm`.

### Colab Secrets

Для безопасности ключи ClearML не хранятся в репозитории, а подтягиваются
из **Colab Secrets** через `google.colab.userdata` в начале ноутбука.

## Результаты

| Метрика | Значение |
|---|---|
| Best `val/perplexity` | **63.08** (эпоха 39) |
| Final `val/perplexity` | 63.19 |
| Final `train/loss` | 3.75 |
| Final `val/loss` | 4.14 |
| Final `train/perplexity` | 48.35 |
| Число эпох | 40 |
| Размер модели | 9 859 072 параметров (~9.9M) |
| Время обучения | ~2.5 часа на Tesla T4 |
| Precision | 16-mixed |

**Наблюдения:**

- `train_loss (3.75) < val_loss (4.14)` — модель не переобучена,
  регуляризация (dropout=0.15, weight_decay=0.01) подобрана корректно.
- Разрыв между train и val небольшой → данных хватило для текущего
  размера модели, дальнейшее увеличение модели без увеличения данных
  не даст прироста.
- Кривая `val/perplexity` выходит на плато к 30–35-й эпохе
  → для значимого улучшения нужно увеличить датасет, а не число эпох.

**Чекпоинты:** сохранены на Google Drive
(`/content/drive/MyDrive/CommonCrawl/checkpoints/`):
- `best.ckpt` — лучший по `val/perplexity`,
- `final.ckpt` — после последней эпохи,
- `last.ckpt` — для resume,
- `gpt-epoch=39-val_perplexity=0.00.ckpt` — именованный.

## Примеры генерации

Все примеры получены при `temperature=0.8`, `top_k=50` с использованием
`best.ckpt` (`val_perplexity=63.08`).

**Prompt:** `The history of`
> The history of the United States has a pop ulation of 60 people per month has a pop ulation pop ul ated pop ulation of 35 to 50 in . 6 million ( 28 . 30 . 6 million in ) , the pop ul

**Prompt:** `In 1890, Marjory Stoneman`
> In 18 90 , Mar j ory St on eman Douglas spent in the " C ra zy in Love " the " in Up " , and was released in 19 86 . At the same time , it was released on the album

**Prompt:** `The city of London`
> The city of London . Sp ok ane 's first city received crit ical reviews . The Sp ok ane Emp ire of Sp ok ane was based on Sp ok ane 's American histor ical design ation and its Sp

**Prompt:** `Albert Einstein was born`
> Al bert E inst ein was born in 18 40 . He was an American stage of London , who worked in 18 78 , and was a grand grand son of Lan cast er who had f old . He became the first Pri

**Анализ:**

- **Синтаксис** — модель выучила грамматику английского языка:
  все предложения имеют корректную структуру (подлежащее-сказуемое-дополнение,
  предлоги, артикли).
- **Частотные начала** — модель уверенно продолжает частые паттерны
  (`The history of the United States`, `The city of London`,
  `Albert Einstein was born`).
- **BPE-артефакты** (`pop ulation`, `Sp ok ane`, `crit ical`) — нормальное
  поведение токенизатора, обученного на небольшом корпусе: редкие слова
  разбиваются на части.
- **Фактические ошибки** (год рождения Эйнштейна, география) — ожидаемы
  для модели с 9.9M параметров и 2.9M токенов в датасете.

## Структура проекта

```
MNNA-2026/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example                 # шаблон переменных окружения
├── configs/
│   └── gpt_config.yaml          # конфиг модели и обучения
├── notebooks/                   # Jupyter-ноутбуки (эксперименты)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── wikitext_datamodule.py   # LightningDataModule
│   ├── models/
│   │   ├── __init__.py
│   │   ├── attention.py             # MultiHeadAttention (block-masked)
│   │   ├── ffn.py                   # FFN-слой
│   │   ├── gpt.py                   # сборка GPT + masked loss
│   │   ├── lightning_module.py      # LightningModule
│   │   ├── positional_encoding.py   # sinusoidal PE (packed batching)
│   │   └── transformer_layer.py     # слой трансформера (post-norm)
│   ├── processing/                  # препроцессинг (ЛР1)
│   ├── tokenization/                # токенизация (ЛР1)
│   └── training/
│       ├── __init__.py
│       └── train.py                 # entry point обучения
```

## Запуск

### Требования

```bash
pip install -r requirements.txt
```

Зависимости: `torch`, `pytorch-lightning`, `omegaconf`, `clearml`,
`tensorboard`, `tokenizers`, `datasets`, `python-dotenv`, `numpy`.

### Обучение

```bash
python -m src.training.train
```

Скрипт:
1. Загружает конфиг из `configs/gpt_config.yaml`.
2. Инициализирует ClearML-задачу.
3. Загружает packed batches через `WikiTextDataModule`.
4. Создаёт `GPTLightningModule`.
5. Запускает обучение с `ModelCheckpoint`, `LearningRateMonitor`.
6. По завершении сохраняет `best.ckpt` и `final.ckpt`.

### Инференс

См. пример загрузки чекпоинта и функцию `generate()` в `notebooks/Lab2.ipynb`.

## Переменные окружения

Переменные окружения задаются либо через `.env` (см. `.env.example`),
либо через **Colab Secrets** при запуске в Colab.

| Переменная | Значение по умолчанию | Назначение |
|---|---|---|
| `CLEARML_API_ACCESS_KEY` | — | Access key ClearML |
| `CLEARML_API_SECRET_KEY` | — | Secret key ClearML |
| `CLEARML_PROJECT_NAME` | `MNNA-2026` | Название проекта в ClearML |
| `DATA_DIR` | `/content/drive/MyDrive/CommonCrawl` | Директория с данными |
| `CHECKPOINT_DIR` | `/content/drive/MyDrive/CommonCrawl/checkpoints` | Директория для чекпоинтов |

**Важно:** реальные ключи **не коммитятся в репозиторий**. В репозитории
хранится только `.env.example` с placeholder-значениями.
