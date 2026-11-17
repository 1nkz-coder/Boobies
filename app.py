# BreastScan Twin

AI decision support for breast cancer, with three trained models in one web app. Every result comes with an explanation, a suggested **BI-RADS / pathology / cytology category** and a **referral pathway** you can print as a report.

| Module | What you give it | Model | Held-out test result |
|---|---|---|---|
| **Scan Lab: Mammogram** | Full mammogram (CC/MLO), PNG/JPG/TIFF/DICOM | EfficientNet-B0 trained on CBIS-DDSM (3,103 images) | AUC **0.758** · sens. 63% · spec. 75% (645 test images) |
| **Scan Lab: Histology** | H&E tissue image (a 50×50 patch or a larger region) | EfficientNet-B0 trained on 277k IDC patches | AUC **0.949** · sens. 89% · spec. 87% (13,500 patches, unseen patients) |
| **Digital Twin** | 30 fine-needle-aspirate measurements (sliders) | RBF-SVM, best of 8 algorithms (WDBC) | Accuracy **97.7%** · AUC 0.999 (171 test cases) |
| Scan Lab: MRI / ultrasound / other | Any breast image | Classic image analysis only (no model trained) | n/a |

> ⚠️ Research and education only. Not a medical device. BI-RADS categories and referrals are **suggestions**; the final assessment must be made by a qualified radiologist or pathologist.

## Run it in VS Code

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** (or press **F5** → *Run BreastScan Twin*).
The console should print `mammogram: ready` and `histopathology: ready`.

## Using the Scan Lab

1. Pick the scan type: **Mammogram**, **Histology slide**, or **MRI / ultrasound / other**.
2. Optional: **Case details** (patient ID, age, side, view, notes). They are only used for the printable report.
3. Drop an image. You get:
   - **Verdict and score**, with the model's decision threshold marked on the bar.
   - **Heatmap** of where the model looked (class-activation map). For histology, each tissue patch is coloured by its IDC probability.
   - **Viewer tools**: zoom (scroll, `+`/`−`), pan (drag), heatmap opacity, brightness, contrast, invert (`I`), toggle heatmap (`H`), compare slider.
   - **Suggested category and referral**: urgency, timeframe, who to refer to, and next steps.
   - **Generate report**: an A4 report (print or save as PDF) with images, category, recommendation and signature lines.
4. Earlier scans from this session appear at the bottom; click one to reopen it.

## How the BI-RADS category is chosen

BI-RADS (ACR, 5th edition) has categories 0–6, each with a likelihood of cancer and a standard management step. The **Learn** page has the full table.

The mammogram model was trained only on CBIS-DDSM images that *contain a finding*, so its score is **not** a calibrated probability of cancer. The official percentage bands (e.g. 4A = 2–10%) therefore can't be applied to it directly. Instead the cut-offs are **anchored on the model's validated decision threshold** (38.1%, chosen on the validation set):

| Model score | Suggested category | Referral |
|---|---|---|
| < ½ × threshold | BI-RADS 2 (benign) | Routine screening |
| ½ × threshold – threshold | BI-RADS 3 (probably benign) | 6-month follow-up imaging |
| threshold → 100%, split into 25 / 30 / 30 / 15 % of the remaining range | 4A / 4B / 4C / 5 | Biopsy and breast clinic → urgent → MDT |
| Image unusable | BI-RADS 0 (incomplete) | Recall for more imaging |

The model never suggests BI-RADS 1, because it never saw normal studies. Histology results use a **pathology pathway** (IDC+ → pathologist confirmation → MDT; once a malignancy is proven the imaging category becomes BI-RADS 6). The Digital Twin uses a **cytology C2–C5 pathway**. All rules live in `core/birads.py` if you want to adapt them to your local guidelines.

## Retraining the models

`kaggle/breastscan_kaggle_train.ipynb` trains everything on Kaggle (GPU T4, Internet on, about 1–1.5 h) and produces `breastscan_models.zip`. Unzip it into `models/`:

```
models/tabular_model.joblib, tabular_metrics.json
models/mammogram/model.onnx, meta.json, training_report.png ...
models/histopathology/model.onnx, meta.json, training_report.png ...
```

The app picks up new model files without a restart. If your local scikit-learn version differs from Kaggle's, the Digital Twin is rebuilt locally with the same recipe, which takes about a second and gives identical results.

For MRI or ultrasound: put labelled images in `data/scans/benign/` and `data/scans/malignant/`, then run `python train_scan_model.py --modality MRI`.

## Project layout

```
app.py                     Flask server + JSON API
core/deep.py               ONNX mammogram / histology inference, preprocessing, heatmaps
core/birads.py             BI-RADS / pathology / cytology categories and referral pathways
core/tabular.py            Digital Twin (SVM): probability, drivers, neighbours, PCA
core/scan.py               classic image analysis (enhancement, segmentation, morphometrics)
templates/, static/        the web interface (no build step)
kaggle/                    training notebook
models/                    trained models + training charts
original_project/          your original notebook and Streamlit app
```

## API

- `POST /api/scan` — multipart `file`, `kind` = mammogram | histopathology | generic. Returns verdict, heatmap layers and `assessment` (category + referral).
- `POST /api/predict` — `{"values": [30 numbers]}`. Returns Digital Twin probability, drivers and the cytology pathway.
- `GET /api/models`, `/api/metrics`, `/api/birads`, `/api/schema`, `/api/case/<i>`

Author: Nana Kofi · nanakofi8430@gmail.com · github.com/1nkz-coder
