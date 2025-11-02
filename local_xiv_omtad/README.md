# XIV: OMTAD Analysis Kit

## What This Is (XIV Acceptance)
This folder delivers the XIV acceptance flow end-to-end: flatten the OMTAD archive, engineer track-level features, train two RandomForest classifiers (vessel type + fishing), and export a demo-ready bundle. Refer to the supporting docs for briefing details:
- [./docs/XIV_DEMO_RUNSHEET.md](./docs/XIV_DEMO_RUNSHEET.md)
- [./docs/XIV_PAPER_ALIGNMENT.md](./docs/XIV_PAPER_ALIGNMENT.md)
- [./docs/XIV_OPERATOR_CARD.md](./docs/XIV_OPERATOR_CARD.md)

## Folder Layout (local-only)
```
local_xiv_omtad/
  data/                # put OMTAD here (zip or extracted)
    source/            # (optional) OMTAD-main/ if extracted
    stage/             # generated flat + features CSVs
  deliverables/omtad_xiv/  # models, charts, metrics, samples for demo
  docs/               # run-sheet, paper alignment, operator card
    refs/             # paper reference notes
  util/               # helpers
```

## Quick Start (macOS / Windows)
```
./setup_env.sh          # macOS/Linux
./setup_env.ps1         # Windows PowerShell
```
Place the dataset at `./data/omtad/OMTAD-main.zip` **or** extract it to `./data/omtad/source/OMTAD-main/`.

### Commands to Satisfy XIV
```
# from repo root
cd local_xiv_omtad
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows

# 1) Flatten
python flatten_omtad.py --source ./data/omtad --out ./data/stage/omtad_flat.csv

# 2) Features
python build_features.py --in ./data/stage/omtad_flat.csv --out ./data/stage/track_features.csv

# 3) Train models (writes pickles + charts + metrics)
python train_models.py --in ./data/stage/track_features.csv --out ./deliverables/omtad_xiv/models

# 4) Create demo track list & bundle
python make_demo_tracks.py
bash ./export_for_demo.sh   # or .\export_for_demo.ps1
python verify_artifacts.py
deactivate
```

## Outputs
- Models: `./deliverables/omtad_xiv/models/*.pkl` (+ `.meta.json`)
- Charts: `./deliverables/omtad_xiv/charts/*.png`
- Metrics: `./deliverables/omtad_xiv/metrics/report.txt`
- Samples: `./deliverables/omtad_xiv/samples/demo_tracks.csv` & `track_features_sample.csv`
- Demo notes: `./DEMO.md`

## Metrics (current run)
- Multiclass macro-F1 = **0.480**
- Fishing vs Rest ROC-AUC = **0.913**

## Demo Without Slides
Upload both `.pkl` files on the Machine Learning page, pick a `track_id` from `deliverables/omtad_xiv/samples/demo_tracks.csv`, click **Classify**, and show the predicted class with probability. Use [./docs/XIV_DEMO_RUNSHEET.md](./docs/XIV_DEMO_RUNSHEET.md) for the exact click path and fallback plan.

## Paper Tie-back & Future Plan
We mirror the paper’s insights (class balance, speed profiles, track geometry) and outline next steps in [./docs/XIV_PAPER_ALIGNMENT.md](./docs/XIV_PAPER_ALIGNMENT.md).

## ✅ Done-when Checklist
- `./data/stage/omtad_flat.csv` exists.
- `./data/stage/track_features.csv` exists.
- `.pkl` models + `.meta.json` written under `./deliverables/omtad_xiv/models/`.
- Charts + `metrics/report.txt` present.
- `samples/demo_tracks.csv` created.
- `python verify_artifacts.py` exits successfully.

## Notes
- Large artefacts are ignored by git; keep everything local to this directory for dev and exports.
- No infrastructure changes required—runs cleanly on macOS or Windows.
