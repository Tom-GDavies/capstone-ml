# XIV Paper Alignment

## Summary
- Captured class distribution summaries via flattened OMTAD dataset (cargo, fishing, passenger, tanker).
- Analysed speed behaviour by class through `speed_hist_by_class.png`.
- Derived track geometry features (distance, turn-rate, duration) to mirror paper insights.
- Delivered per-track feature CSV enabling further statistical comparisons.
- Exported demo bundle providing artefacts analogous to the paper’s exploratory figures.
- See `./refs/OMTAD_PAPER_REF.md` for paper reference context.

## Results
- Multiclass vessel-type random forest macro-F1: **0.480**.
- Fishing vs non-fishing ROC-AUC: **0.913**.
- Implication: baseline models surface meaningful separation while leaving room for refinement.

## Limitations
- Workflow reimagines the study; not a line-for-line replication, but sufficient for demo narratives.

## Further Insights Plan
- Integrate Tom’s anomaly detection for unusual tracks.
- Layer seasonality maps over lat/lon bins.
- Examine near-shore and port dwell times.
- Add tortuosity and dwell percentage metrics.
- Expand with temporal cross-validation to capture yearly drift.

## Reproduce
1. `python flatten_omtad.py --source ./data/omtad --out ./data/stage/omtad_flat.csv`
2. `python build_features.py --in ./data/stage/omtad_flat.csv --out ./data/stage/track_features.csv`
3. `python train_models.py --in ./data/stage/track_features.csv --out ./deliverables/omtad_xiv/models`
4. `./export_for_demo.sh` (or PowerShell equivalent)
