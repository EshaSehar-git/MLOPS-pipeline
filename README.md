# Invoice/Receipt Extraction MLOps Pipeline

End-to-end pipeline: image/PDF upload → OCR → structured field extraction →
confidence validation → API serving → drift monitoring.

## Project structure

```
invoice-extraction-mlops/
├── data/
│   ├── raw/                    # put the raw SROIE dataset here
│   ├── processed/              # train/val/test splits, converted formats
│   └── sample_receipts/        # a few images for quick manual testing
├── src/
│   ├── ocr/extract_text.py         # OCR wrapper (Tesseract / EasyOCR)
│   ├── extraction/field_parser.py  # rule-based baseline extractor (no training needed)
│   ├── validation/confidence_scorer.py
│   ├── training/
│   │   ├── prepare_dataset.py      # converts SROIE -> LayoutLMv3 token-classification format
│   │   └── train_layoutlmv3.py     # fine-tuning script (run in Colab)
│   └── monitoring/drift_check.py   # Evidently AI drift report
├── api/
│   ├── main.py                 # FastAPI app
│   └── schemas.py
├── notebooks/
│   └── colab_train_layoutlmv3.ipynb  # (see COLAB SECTION below - just a script here, paste into a notebook)
├── tests/test_pipeline.py
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── requirements.txt
└── README.md
```

## Two extraction approaches in this repo

You get **both**, so you can start simple and upgrade:

1. **Rule-based baseline** (`src/extraction/field_parser.py`) — regex over raw OCR
   text. Zero training required. Good enough to get the whole pipeline (API,
   Docker, CI/CD, monitoring) working on day one.
2. **LayoutLMv3 fine-tuned model** (`src/training/train_layoutlmv3.py`) — the
   real ML upgrade. Trained on SROIE. Swap it into the API once trained.

## Why LayoutLMv3?

For document field extraction (as opposed to plain text classification),
**layout matters as much as the words themselves** — "TOTAL" and the number
next to it are related because of their *position* on the receipt, not
because of surrounding sentence grammar. LayoutLMv3 is a good fit because:

- It's a **multimodal transformer**: it jointly encodes text, the 2D position
  of each word (bounding boxes), and the image itself — so it understands
  that a number in the bottom-right near the word "Total" is probably the
  total amount, the way a human visually parses a receipt.
- It's pretrained on millions of scanned documents, so fine-tuning on ~600
  labeled SROIE receipts is enough to get strong results — you're not
  training from scratch.
- It's framed as a **token classification** problem (label each OCR word as
  `O`, `B-COMPANY`, `B-DATE`, `B-TOTAL`, `B-ADDRESS`, etc.), which maps
  directly onto what SROIE's ground truth already gives you.
- It's free and open on Hugging Face (`microsoft/layoutlmv3-base`), and small
  enough to fine-tune on a single free Colab GPU (T4).

Alternative you could use instead: **Donut** (OCR-free document
understanding transformer) — skips the separate OCR step entirely and reads
the image directly. More elegant, but heavier to train and harder to debug
when it's wrong. LayoutLMv3 is the better learning project because you can
inspect each pipeline stage (OCR output → tokens → labels) independently.

## Setup (local)

```bash
git clone <your-repo>
cd invoice-extraction-mlops
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# system dependency for Tesseract OCR
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr
# Mac:
brew install tesseract
```

Place the SROIE dataset under `data/raw/` so you have:
```
data/raw/task1_ocr/*.jpg + *.txt
data/raw/task2_key_info/*.jpg + *.txt
```
(adjust `prepare_dataset.py` paths if your download uses different folder names)

## Run the rule-based pipeline locally (no training needed)

```bash
python src/ocr/extract_text.py --image data/sample_receipts/receipt1.jpg
python src/extraction/field_parser.py --image data/sample_receipts/receipt1.jpg
```

## Run the API

```bash
uvicorn api.main:app --reload --port 8000
# then POST an image to http://localhost:8000/extract
```

## Docker

```bash
docker build -t invoice-extraction .
docker run -p 8000:8000 invoice-extraction
```

---

## TRAINING ON GOOGLE COLAB (LayoutLMv3)

### 1. Open a new Colab notebook
Go to https://colab.research.google.com → New Notebook

### 2. Turn on the free GPU
`Runtime` → `Change runtime type` → `Hardware accelerator` → `T4 GPU` → Save

### 3. Cell 1 — install dependencies
```python
!pip install -q transformers datasets seqeval accelerate pytesseract
!apt-get install -y tesseract-ocr
```

### 4. Cell 2 — upload the SROIE dataset
Easiest path: upload the dataset to your Google Drive first, then mount it.
```python
from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = "/content/drive/MyDrive/SROIE"  # adjust to wherever you put it
```
(Alternative: `from google.colab import files; files.upload()` to upload a
zip directly, then `!unzip sroie.zip -d /content/data`)

### 5. Cell 3 — clone your repo or paste the training scripts
```python
!git clone https://github.com/<your-username>/invoice-extraction-mlops.git
%cd invoice-extraction-mlops
```
(Or just copy `src/training/prepare_dataset.py` and
`src/training/train_layoutlmv3.py` into Colab cells directly if you don't
want to push to GitHub yet.)

### 6. Cell 4 — prepare the dataset
```python
!python src/training/prepare_dataset.py \
    --raw_dir "$DATA_DIR" \
    --output_dir /content/processed
```

### 7. Cell 5 — run training
```python
!python src/training/train_layoutlmv3.py \
    --data_dir /content/processed \
    --output_dir /content/drive/MyDrive/layoutlmv3-sroie-model \
    --epochs 15 \
    --batch_size 4 \
    --lr 5e-5
```

### 8. Cell 6 — track the run with MLflow (optional but recommended)
```python
!pip install -q mlflow
import mlflow
mlflow.set_tracking_uri("file:/content/drive/MyDrive/mlruns")
mlflow.set_experiment("layoutlmv3-sroie")
# train_layoutlmv3.py already logs params/metrics if MLflow is installed —
# see the script for the mlflow.start_run() block
```

### 9. Download the trained model back to your machine
Since you saved it to `/content/drive/MyDrive/layoutlmv3-sroie-model`, it's
already in your Google Drive — just download that folder, or zip it first:
```python
!zip -r /content/drive/MyDrive/layoutlmv3-sroie-model.zip /content/drive/MyDrive/layoutlmv3-sroie-model
```

### 10. Load it back into the FastAPI app
Drop the downloaded model folder into `models/layoutlmv3-sroie/` locally and
point `api/main.py`'s model loader at that path (see comment in
`api/main.py`).

---

## Monitoring

After you have real predictions flowing in, run:
```bash
python src/monitoring/drift_check.py --reference data/processed/train.csv --current data/processed/new_batch.csv
```
This generates an Evidently HTML report showing whether new receipts look
statistically different from your training data (e.g., a new template broke
your extraction).
