# CIFAKE Web App

A Flask front-end for the existing CIFAKE CNN pipeline. **No ML code was changed** —
`app.py` only imports and calls the functions that already existed in `src/`:

| Existing function (untouched) | Used for |
|---|---|
| `src.model.load_trained_model` | Loading the trained `.keras` model once at server startup |
| `src.predict.predict_single_image` | REAL/FAKE label + confidence + probabilities |
| `src.gradcam.explain_image` | Grad-CAM heatmap overlay generation |
| `src.preprocessing.preprocess_single_image` | Used internally by the two functions above |

## Setup

```bash
pip install -r requirements.txt
```

Make sure your trained model exists at:

```
outputs/models/cifake_final_model.keras
```

(this is unchanged from the CLI workflow — train with `python main.py train` if needed).

## Run

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

## Pages

| Route | Description |
|---|---|
| `/` | Landing page |
| `/upload` | Drag-and-drop / file-picker upload form |
| `/predict` (POST) | Saves the upload, runs prediction + Grad-CAM, stores a history row, redirects to the result |
| `/result/<id>` | Verdict, confidence, probability bars, Grad-CAM side-by-side |
| `/history` | All past predictions (SQLite), newest first |
| `/report/<id>` | Streams a generated PDF report for that prediction |

## New files added (nothing existing was modified)

```
app.py                      # Flask entry point
database/db.py               # SQLite helper (predictions table)
database/history.db          # created automatically on first run
templates/                   # Jinja2 pages (base, index, upload, result, history, 404)
static/css/style.css         # Dark/light "forensic scan" theme
static/js/main.js            # Theme toggle, drag-and-drop, scroll reveal, progress bars
static/uploads/              # Temporary storage for uploaded images
static/gradcam/              # Grad-CAM overlays (src/gradcam.py's existing output_dir param
                              # is simply pointed here instead of outputs/gradcam/)
```

Your original CLI workflow (`python main.py predict --image ...`, `python main.py gradcam ...`,
`python main.py train`, `python main.py evaluate`) still works exactly as before — `src/` and
`main.py` were not touched.
