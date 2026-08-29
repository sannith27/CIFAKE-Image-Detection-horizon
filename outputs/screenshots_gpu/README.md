# GPU Run Evidence and Captions

All artifacts in this folder were generated after the completed WSL2 CUDA run on the NVIDIA GeForce RTX 3050 6GB Laptop GPU.

**Figure 1.** `01_dataset_train_real_fake_samples.png` — Three REAL and three FAKE samples copied from `dataset/train` for visual reference.

**Figure 2.** `accuracy_loss_curves.png` — Accuracy and loss curves from the completed full-dataset GPU training run.

**Figure 3.** `confusion_matrix.png` — Confusion matrix from full 20,000-image test evaluation using `python main.py evaluate` under the GPU runtime.

**Figure 4.** `roc_curve.png` — ROC curve from the full GPU-backed test evaluation, with ROC-AUC 0.9821.

**Figure 5.** `precision_recall_curve.png` — Precision-recall curve from the full GPU-backed test evaluation.

**Figure 6.** `06_gradcam_real.png` — GPU-generated Grad-CAM overlay for `train/REAL/0000 (10).jpg`, predicted REAL at 98.86% confidence.

**Figure 7.** `07_gradcam_fake.png` — GPU-generated Grad-CAM overlay for `train/FAKE/1000 (10).jpg`, predicted FAKE at 98.73% confidence.

**Figure 8.** `02_upload.png` — Web application upload page supporting BMP, GIF, JPG, JPEG, PNG, and WEBP user images.

**Figure 9.** `03_result_real.png` — Web result page showing verdict, confidence, probability bars, original image, and Grad-CAM overlay.

**Figure 10.** `04_history.png` — Web history page containing REAL, FAKE, and user-supplied image predictions.

**Figure 11.** `05_report_real.pdf` — PDF report generated from `/report/8` for the GPU-backed REAL prediction.

## GPU run metrics

- Training data: 100,000 images, all used.
- Validation data: 20,000 images, all used.
- Test data: 20,000 images, all used.
- Training: 1 complete epoch, batch size 64, GPU `/device:GPU:0`.
- Test accuracy: 93.15%.
- Test precision: 92.50%.
- Test recall: 93.90%.
- Test F1: 93.20%.
- Test ROC-AUC: 98.21%.
