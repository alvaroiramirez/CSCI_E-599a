# Disaster Damage Detection Using LLMs and Aerial Imagery

This project uses OpenAI's GPT-4o Vision API to detect and classify disaster-related damages in aerial imagery from the LADI v2 dataset. It consists of two core scripts:


---

## Configuration

Both scripts define constants at the beginning that control their behavior:

### `LLM_Captioning_Raw.py` constants:
- `CALL_OPENAI_API`: Set to `True` to make real API calls, or `False` to run the program without calling the API.
- `SOURCE_FOLDER`: Path to the project folder (e.g., `LADIv2`).
- `API_KEY_FILE`: Path to the file containing your OpenAI API key.
- `MODEL`: OpenAI model to use (e.g., `gpt-4o`).
- `QUALITY_SEAL_DARK_THRESHOLD`: Brightness threshold to flag dark images.
- `QUALITY_SEAL_BLUR_THRESHOLD`: Blur threshold for flagging blurry images.
- `API_REQUEST_DELAY`: Delay (in seconds) between requests to avoid rate limiting.
- `COST_PER_TOKEN`: Token pricing rate (used for cost estimation).

### `Captioning_Assessment.py` constants:
- `COST_PER_TOKEN`: Token pricing rate used for cost summary.
- `DATASET_FOLDER`: Specifies the name of the folder containing results and images.
- Paths to `results.csv`, `analysis.json`, and `LADIv2.csv` are constructed from this base.

Adjust these constants as needed to reflect your local folder structure, OpenAI pricing, and processing preferences.

---

## 1. `LLM_Captioning_Raw.py`

This script analyzes individual images and generates binary results for various types of damages.

### Features:
- Sends aerial images to the GPT-4o Vision model.
- Extracts descriptive captions and binary indicators for:
  - Buildings, roads, bridges, trees, water, flooding, debris
- Flags blurry or dark images using configurable thresholds.
- Supports API call toggling and checkpointing to avoid reprocessing.
- Categorizes images into folders based on classification.
- Outputs:
  - `analysis.json` — detailed metadata per image
  - `results.csv` — per-image binary results for downstream analysis

---

## 2. `Captioning_Assessment.py`

This script evaluates the predictions produced by `LLM_Captioning_Raw.py` against the LADI v2 ground truth.

### Features:
- Merges predictions with `LADIv2.csv`.
- Calculates:
  - Accuracy per category
  - TP, TN, FP, FN
  - Precision, Recall, F1-score
  - Exact Match Ratio
  - Hamming Loss
- Builds an Excel report with:
  - **Comparison** tab: True vs Predicted vs Match
  - **Metrics** tab: Per-category classification metrics
  - **Summary** tab:
    - Token usage and cost per subfolder
    - Session start/end timestamps
    - Total duration (formatted)

### Console Output:
- Summary of all metrics
- Per-category accuracy
- Tabular summary of token/cost by subfolder

---

## Folder Structure

```
project_root/
│
├── 4.LLM/
│   ├── LLM_Captioning_Raw.py
│   ├── Captioning_Assessment.py
│   └── README.md
│
├── LADIv2/
│   ├── data/                # Input images
│   │   ├── train
│   │   ├── validation
│   │   └── test
│   └── results/             # Outputs
│       ├── needs_review    (folder)
│       ├── no_damages      (folder)
│       ├── not_discernible (folder)
│       ├── with_damages    (folder)
│       ├── results.csv
│       ├── analysis.json
│       └── comparison.xlsx
│   
└── LADIv2.csv               # Ground truth
```

---

## Requirements

- Python 3.9+
- `openai`
- `openpyxl`
- `pandas`
- `rich`
- `scikit-learn`
- `Pillow`
- `opencv-python`
- `numpy`

---

## How to Use

1. Copy the images to be processed to the `LADIv2/data` folder. The script will automatically detect the subfolders and process all images within `LADIv2/data` and its subfolders.

2. Analyze images with:
```bash
python LLM_Captioning_Raw.py
```

3. Evaluate results and generate reports:
```bash
python Captioning_Assessment.py
```

4. Review classification 
  `no_damages`      folder for images with no damages.
  `with_damages`    folder for images with damages.
  `not_discernible` folder for images that are not discernible.
  `needs_review`    folder for images that need manual review.
  
5. Open 
   `results.csv` for binary results per category per image. Includes timestamp.
   `analysis.json` for detailed metadata per image. Includes cost information.
   `comparison.xlsx` for a detailed Excel report.

---

## Notes

- API key must be present at the path defined in `API_KEY_FILE`. Adjust path as needed.
- You can resume interrupted processes by restarting the script.
- Modify the `SOURCE_FOLDER` and `DATASET_FOLDER` constants to work with other image folders.
