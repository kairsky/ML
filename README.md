# University Machine Learning & Deep Learning Project

Полный учебный проект: классическое ML, снижение размерности, MLP, CNN, трансферное обучение, отчёт PDF, демо.

## Структура сдачи (Required Files)

| Файл / папка | Описание |
|--------------|----------|
| **`report.pdf`** | Итоговый отчёт (все 6 задач, ≥20 страниц) |
| **`notebooks/`** | Jupyter для задач 2, 3, 4, 5 |
| **`src/`** | Модульный Python-код |
| **`results/`** | Модели, графики, метрики, логи |
| **`README.md`** | Инструкция по запуску |
| **`requirements.txt`** | Зависимости |
| **`demo.py`** | Демонстрация работы для защиты |

## Быстрый старт

```powershell
cd "ML Assik"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Сгенерировать report.pdf и блокноты
python scripts\build_report_pdf.py
python scripts\generate_notebooks.py

# Обучение (полное)
python scripts\run_all.py

# Быстрая проверка
python scripts\run_all.py --quick

# Демо для показа
python demo.py
```

## Архив для сдачи

```powershell
python scripts\build_report_pdf.py
python scripts\package_submission.py --id YOUR_GROUP_ID
# → ProjectGroup_YOUR_GROUP_ID.zip
```

## Задачи проекта

| Задача | Содержание | Артефакты |
|--------|------------|-----------|
| 1 | Теория классических алгоритмов | `docs/report/sections/task1_theory.md`, `report.pdf` |
| 2 | Classical ML pipeline | `notebooks/02_*`, `results/models/classical_ml/` |
| 3 | PCA, t-SNE, UMAP, LDA | `notebooks/03_*`, `results/plots/dimensionality/` |
| 4 | MLP Fashion-MNIST | `notebooks/04_*`, `results/plots/mlp/` |
| 5 | CNN + ResNet18 | `notebooks/05_*`, `results/models/cnn/` |
| 6 | Survey + сравнение | `docs/report/sections/task6_survey.md` |

## Демонстрация (`demo.py`)

Скрипт показывает:
- загрузку данных и inference сохранённой модели;
- таблицу метрик;
- пути к графикам для презентации.

## Отчёт PDF

```powershell
python scripts\build_report_pdf.py
```

Создаёт:
- `docs/report/full_report.md` — полный текст (~5000+ слов)
- `report.pdf` — PDF с рисунками из `results/plots/`

## Репозиторий

https://github.com/kairsky/ML

## Структура каталогов

```
ML Assik/
├── report.pdf
├── demo.py
├── README.md
├── requirements.txt
├── config/config.yaml
├── src/
├── scripts/
├── notebooks/
│   ├── 02_classical_ml_pipeline.ipynb
│   ├── 03_dimensionality_reduction.ipynb
│   ├── 04_mlp_experiments.ipynb
│   └── 05_cnn_and_transfer_learning.ipynb
├── docs/report/sections/    # исходники отчёта
└── results/
    ├── models/
    ├── plots/
    └── metrics/
```
