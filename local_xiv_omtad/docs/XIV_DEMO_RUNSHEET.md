# XIV Demo Run Sheet

## Pre-demo Checks
- Confirm `../deliverables/omtad_xiv/` exists on disk.
- Validate `../deliverables/omtad_xiv/models/*.pkl` are present.
- Ensure `../deliverables/omtad_xiv/samples/demo_tracks.csv` is available for quick lookups.
- Verify the demo web app is reachable and credentials work.

## Upload Models (Machine Learning Page)
1. Navigate to **Machine Learning** in the UI.
2. Use the upload control to select `../deliverables/omtad_xiv/models/omtad_vessel_type.pkl`.
3. Repeat for `../deliverables/omtad_xiv/models/omtad_fishing.pkl`.
4. Confirm both appear in the model selector list.

## Classify Flow
1. Open `../deliverables/omtad_xiv/samples/demo_tracks.csv`.
2. Copy any `track_id` from the sheet.
3. In the UI, paste the `track_id`, click **Classify**, and read back the predicted vessel class and probability.

## Talk Track
- Dataset: OMTAD West Grid vessel tracks (2018–2020) prepared for local analysis.
- Features: track-level summary (point count, duration, mean/max speed, turn-rate, distance).
- Metrics: Multiclass macro-F1 **0.480**; fishing binary ROC-AUC **0.913**.
- Further insights: use anomaly and seasonality exploration (reference Tom’s ongoing work).

## Fallback
If uploads fail, screen-share `../deliverables/omtad_xiv/charts/*.png` and walk through `../deliverables/omtad_xiv/metrics/report.txt` to discuss results.

## ✅ Done When
- Models uploaded and selectable.
- A track_id from `demo_tracks.csv` successfully classified with probabilities shown.
- Charts and metrics ready as backup material.
