
# Real-Time Speech Command Recognition with Deep Learning  

## Project Overview  
This project implements a **real-time speech command recognition system** using **Deep Speech Convolutional Neural Network (DS-CNN)**. The model is trained on audio clips of spoken words, converts them into **Mel spectrograms with delta & delta-delta features**, and predicts commands from live microphone input.  

The pipeline includes:  
- Data preprocessing (waveform → spectrogram)  
- Model training with advanced techniques (AdamW, CosineAnnealingLR, label smoothing)  
- Evaluation with F1-score, confusion matrix, and plots  
- Real-time inference with microphone recording  

---

## Features  
* Converts raw audio into **log-mel spectrograms** with delta features  
* **DS-CNN architecture** optimized for keyword spotting (KWS)  
* Training with **label smoothing** and **CosineAnnealing scheduler**  
* Evaluation with **confusion matrix, precision, recall, F1**  
* Real-time inference with **mic recording in Google Colab**  
* Silence detection using **energy-based threshold**  
* Visualization of **waveform & spectrogram** for debugging  

---

## Repository Structure  
```
├── Notebooks/
│   ├── 1_Training.ipynb         # Training pipeline (preprocessing, training, evaluation)
│   ├── 2_Inference.ipynb        # Load best model & run live predictions
│
├── Artifacts/
│   ├── best_by_val_f1.pt        # Best model checkpoint (saved during training)
│   ├── final_model.pth          # Final trained weights
│
├── Plots/
│   ├── training_curves.png      # Loss/accuracy plots
│   ├── confusion_matrix_test.png # Confusion matrix on test set
│
├── README.md                    # Project documentation (this file)
```

---

## Model Architecture (DS-CNN for KWS)  
The **DS-CNN** model is designed for **efficient speech command recognition**:  

- Input: `[3, 40, T]` (logmel + delta + delta² features)  
- Depthwise-separable convolutions  
- Batch normalization & ReLU  
- Global average pooling  
- Fully connected layer → softmax over classes  

This makes it lightweight yet powerful for **real-time inference**.

---

## Setup & Installation  

### 1. Clone the repo  
```bash
git clone https://github.com/DaivikPatel0/speech-command-recognition.git
cd speech-command-recognition
```

### 2. Install dependencies  
```bash
pip install torch torchaudio matplotlib numpy
```

If using **Google Colab**, most dependencies are pre-installed.

---

## Training  

Run the training notebook:  

```bash
notebooks/1_training.ipynb
```

- Loads dataset (waveforms → log-mel spectrograms)  
- Trains DS-CNN for **35 epochs** with:  
  - Optimizer: **AdamW**  
  - Scheduler: **CosineAnnealingLR**  
  - Loss: **CrossEntropy with Label Smoothing (0.05)**  
- Saves:  
  - `best_by_val_f1.pt` (best model)  
  - `final_model.pth` (last epoch model)  
- Outputs training curves + confusion matrix  

---

## Inference (Live Mic Prediction)  

Run the inference notebook:  

```bash
notebooks/2_inference.ipynb
```

1. Loads **best checkpoint** & label mapping  
2. Records **1.2s audio clip** via mic (browser-based in Colab)  
3. Converts audio → logmel + delta + delta² features  
4. Runs prediction with trained DS-CNN  
5. Applies **silence thresholding** to avoid false detections  
6. Displays:  
   - Top predictions with confidence  
   - Waveform & spectrogram plots  

Example output:  

```
File: /content/mic.wav
RMS level: -43.5 dBFS | Silence gate: -50.0 dBFS
Top predictions:
   1. yes                  0.982
   2. no                   0.012
   3. __silence__          0.006
```

---

## Evaluation  
- Accuracy, Precision, Recall, F1-score (macro/micro/weighted)  
- Confusion matrix visualization  
- Training vs validation loss/accuracy curves  

---

## Results  
- Achieved **90.14% accuracy** and **87.28 F1-score** on the test set.  
- Robust silence handling with energy-based gating.  
- Works in **real-time inference** on Colab or local GPU.  

---

## Future Improvements  
- Add **data augmentation** (SpecAugment, noise injection)  
- Deploy model on **mobile/edge devices**  
- Extend to **continuous streaming recognition** (sliding window inference)  
- Experiment with **Transformer-based audio models**  

---

## Acknowledgements  
- [TorchAudio](https://pytorch.org/audio/stable/) for preprocessing functions  
- [Google Speech Commands Dataset](https://www.kaggle.com/datasets/sylkaladin/speech-commands-v2)  
- Inspiration from **KWS research papers & tutorials**  

---

## License  
MIT License — free to use and modify with attribution.  
