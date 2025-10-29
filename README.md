# FusionNet: Deepfake Audio Detection (SpecRNet + RawGAT) with Server-Side Continual Learning & ONNX Mobile Deployment

FusionNet (RawGAT-ST + SpecRNet) for detecting fake vs real speech. Train in PyTorch (Task-A clean, Task-B telephony/continual), export to ONNX, and can be evaluated on external datasets.

Features:
  - Dual-branch FusionNet: raw waveform + Mel-spectrogram fusion.
  - Task A (baseline, clean audio) and Task B (telephony-style continual fine-tune).
  - One-click ONNX evaluator: runs on any folder with real/ and fake/ (case-insensitive) files.
  - Metrics & artifacts: AUC, EER, ROC plot, Confusion Matrix, per-file predictions CSV.

Datasets
Used during development/evaluation :
  - AUDETER subset (fake clips) → used 3500 files for each folder (ie. tts, Hifigan, cosyvoice,..) for training, validation and testing
  - Combined real datasets of ASVspoof 2021 and In-the-Wild audio deepfake → REAL pool used for training, validation and testing
  - The Fake or Real Dataset (external test) → mixed codecs/durations for robustness check using ONNX models.


Model files
  - .pth – full PyTorch weights (used for training/continual learning).
  - .onnx – inference-only graph for deployment/testing for mobile usage (no continual learning).
  - any ONNX models produced (fusionnet_taskB.onnx, fusionnet_taskA.onnx) can be tested by pointing ONNX_PATH to it and running the ONNX quick test section.

# How to run on Kaggle (ONNX quick test)

Steps:-
1. Open the notebook and scroll to the section titled "TEST ON ONNX FILE STARTS # FROM HERE."
2. Click Add Input and attach:
    - ONNX model dataset that downloaded from this git repositories (then set ONNX_PATH to the file inside it).
    - A deepfake-voice dataset organized with two folders: real/ and fake/ (case-insensitive).

3. Set the paths at the top of that section:
      ONNX_PATH = "/kaggle/input/<your-onnx-dataset>/<path>/fusionnet_taskX.onnx"
      DATA_ROOT = "/kaggle/input/<your-audio-dataset>/<root-folder>"

4. Run all cells from that section downward (do not run the training sections).

5. Outputs:-
  predictions_*.csv (per-file probabilities and labels)
  roc_*.png (ROC curve + AUC)
  cm_*.png (confusion matrix @ EER)
  *_report.json (AUC, EER, threshold, confusion matrix)
