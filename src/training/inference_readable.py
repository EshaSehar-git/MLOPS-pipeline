"""
Word-level inference: runs Tesseract OCR ourselves (so we know the exact
words + boxes), feeds them into LayoutLMv3, then merges the model's
sub-word predictions back into whole words with their actual text —
so the output is human-readable instead of raw BPE token fragments.

Run this in Colab (from inside the invoice-extraction-mlops project folder).
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from PIL import Image
import torch
from collections import defaultdict

from src.ocr.extract_text import extract_text  # our own Tesseract wrapper

MODEL_PATH = "/content/drive/MyDrive/layoutlmv3-sroie-model"
IMAGE_PATH = "/content/drive/MyDrive/SROIE2019/test/img/X51005230605.jpg"  # change per test

model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH, apply_ocr=False)
id2label = model.config.id2label

# 1. Run OCR ourselves so we have exact words + boxes
ocr_words = extract_text(IMAGE_PATH, engine="tesseract")
image = Image.open(IMAGE_PATH).convert("RGB")
width, height = image.size

words = [w.text for w in ocr_words]
boxes = [
    [
        int(1000 * w.left / width),
        int(1000 * w.top / height),
        int(1000 * (w.left + w.width) / width),
        int(1000 * (w.top + w.height) / height),
    ]
    for w in ocr_words
]

# 2. Feed known words + boxes into the model
encoding = processor(
    image, words, boxes=boxes, return_tensors="pt", truncation=True
)

with torch.no_grad():
    outputs = model(**encoding)

probs = torch.softmax(outputs.logits, dim=-1).squeeze()
predictions = probs.argmax(-1).tolist()
confidences = probs.max(-1).values.tolist()
word_ids = encoding.word_ids(batch_index=0)

# 3. Merge sub-word predictions back to whole words (first sub-token wins)
seen = set()
word_predictions = {}
for idx, word_id in enumerate(word_ids):
    if word_id is None or word_id in seen:
        continue
    seen.add(word_id)
    label = id2label[predictions[idx]]
    if label != "O":
        word_predictions[word_id] = (label, confidences[idx])

# 4. Group by field and print actual text
fields = defaultdict(list)
for word_id, (label, conf) in word_predictions.items():
    field_name = label.replace("B-", "").replace("I-", "")
    fields[field_name].append((words[word_id], conf))

print(f"=== Predicted fields for {IMAGE_PATH} ===\n")
for field in ["COMPANY", "DATE", "ADDRESS", "TOTAL"]:
    items = fields.get(field, [])
    if not items:
        print(f"{field}: (none found)")
        continue
    text = " ".join(w for w, c in items)
    avg_conf = sum(c for w, c in items) / len(items)
    print(f"{field}: {text}   (avg confidence: {avg_conf:.2f})")

