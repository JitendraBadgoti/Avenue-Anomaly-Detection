# Avenue Anomaly Detection
Author: Jitendra Badgoti (Enrollment: 25117068)

## **Project Vision**
Surveillance anomaly detection often fails due to "identity mapping" (where models simply learn to copy pixels) and real-world data corruption. This Pipeline solves this by implementing a two-stage architecture:

**Stage 1:** The Gatekeeper (Supervised Restoration)

**Stage2:** The U-Net Predictor (Unsupervised Detection)
## **Architecture & Vital Components**
### 1. The "Gatekeeper"

**Model:** ResNet-18.

**Function:** Classifies frames into Normal or Corrupted.

**Vital Logic:** If a frame is flagged as corrupted, the pipeline applies a $180^\circ$ rotation followed by a horizontal flip. This ensures the prediction model maintains spatial consistency.
### 2. Temporal Prediction U-Net
Unlike traditional autoencoders that reconstruct the current frame, this model predicts the future frame.

**Input:** 4 consecutive frames (12 channels).

**Target:** 5th frame prediction (3 channels).

**Skip Connections:** These are vital for preserving high-resolution background details, allowing the model to focus its capacity on moving objects.
### 3. Local Max-Pooling Error Extraction
Standard MSE loss often "smooths out" small anomalies (like a flying bag).

**The Innovation:** During inference, we calculate the L1 error and pass it through a $16 \times 16$ Max-Pooling layer.

**Impact:** This captures the peak deviation in localized areas, making the model highly sensitive to small, fast-moving objects that would otherwise be averaged out in a global score.
## Installation & Usage
1. Requirements
   - pip install -r requirements.txt
     
_Dependencies: PyTorch, Torchvision, Pandas, Scipy, OpenCV, Pillow, Tqdm._

2. Pipeline Execution
     - **Configuration:** Update the directory paths in the CONFIGURATION block.

     - **Training Stage:** Uncomment train_gatekeeper() and train_unet() in the main block.

     - **Inference:** Run run_final_inference() to generate the submission_GOAT_6.csv.

## Final Metrics
Post-Processing: Gaussian Smoothing ($\sigma=1.5$) + Score Squaring ($x^2$) for high contrast.

## 📂 Repository Structure
├── README.md                 
├── Avenue_Anomaly_Detection_Notebook.ipynb    
├── requirements.txt          
