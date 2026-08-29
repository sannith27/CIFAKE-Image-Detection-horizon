# CIFAKE Evidence Screenshots and Captions

These files were generated from the current project outputs and local Flask app.

**Figure 1.** `00_dataset_train_samples.png` — Representative REAL and FAKE images sampled from the training split.

**Figure 2.** `05_accuracy_loss_curves.png` — Training and validation accuracy/loss curves from the existing completed training artifact.

**Figure 3.** `06_confusion_matrix.png` — Confusion matrix produced by the evaluation pipeline on the test split.

**Figure 4.** `07_roc_curve.png` — ROC curve showing the classifier's discrimination across probability thresholds.

**Figure 5.** `08_precision_recall_curve.png` — Precision-recall curve showing the precision/recall tradeoff across thresholds.

**Figure 6.** `09_gradcam_real.png` — Grad-CAM overlay for a REAL prediction, highlighting image regions used by the model.

**Figure 7.** `10_gradcam_fake.png` — Grad-CAM overlay for a FAKE prediction, highlighting image regions used by the model.

**Figure 8.** `01_upload.png` — Flask upload page showing the drag-and-drop image analysis interface.

**Figure 9.** `02_result_real.png` — Flask result page showing the REAL verdict, confidence, probability bars, original image, and Grad-CAM overlay.

**Figure 10.** `03_history.png` — Flask history page listing multiple REAL and FAKE analyses with confidence and timestamps.

**Figure 11.** `04_report_real.pdf` — Generated PDF report containing the filename, timestamp, verdict, probabilities, original image, and Grad-CAM overlay.

## Verification notes

- The four graph files were already present in `outputs/graphs/` before this evidence bundle was assembled.
- The Flask app was exercised with real images from `train/REAL` and `train/FAKE`; prediction records and Grad-CAM files are stored in `database/history.db` and `static/gradcam/`.
- The PDF was downloaded from `/report/2` and verified as a one-page PDF with readable extracted text.
- The latest full retraining attempt stopped before writing new history because TensorBoard was missing; `requirements.txt` now includes `tensorboard>=2.20`, and TensorBoard was installed into the project virtual environment.
