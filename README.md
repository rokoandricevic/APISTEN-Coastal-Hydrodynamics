# APISTEN: Adaptive Physics-Informed Spatio-Temporal Ensemble Network

An advanced, physics-informed deep learning framework optimized for robust, long-form satellite remote sensing inversions of coastal water quality parameters (Chlorophyll-$\alpha$, CDOM, and Turbidity). 

Unlike conventional physics-informed models that enforce rigid, globally weighted physical constraints, APISTEN introduces the structural concept of **runtime algorithmic decoupling**. By separating the data-driven spatio-temporal feature extraction layers (the U-Net ensemble core) from an independent, parallel trust-region gating pathway, the network maintains structural adaptivity across complex environmental regimes.

---

## 🚀 Key Features
* **Runtime Algorithmic Decoupling:** Isolates neural spatial feature maps from physical constraints during runtime inference to prevent gradient tension and noise-driven distortions.
* **Dynamic Trust-Region Scaling:** Utilizes a pixel-by-pixel Truncated Gaussian Umbrella based on real-time out-of-sample spectral standardization, automatically dropping to a protective floor ($\lambda_{\text{min}} = 0.1$) in low-signal regimes.
* **Continuous Seasonal Embedding:** Incorporates a phase-continuous harmonic coordinate ($t_{\text{mod}}$) to natively resolve seasonal bio-optical non-stationarity without requiring fragmented, month-specific calibration sets.
* **Self-Diagnostic Quality Flags:** Deploys a multi-model stochastic ensemble matrix ($M = 5$) to translate uncompensated sensor-level anomalies (e.g., detector striping) into explicit spatial epistemic uncertainty maps ($\sigma$).

---

## 📂 Repository Structure

The core production code consists of the following foundational tracking scripts:

* 📄 **`APISTEN_Time_Series_Pipeline.py`**: The master execution script governing the spatio-temporal convolutional ensemble, out-of-sample standardization tracks, loss optimization loops, and final matrix outputs.

* 📄 **`scan_jugo_anomalies.py`**: The automated meteorological processing track designed to ingest ERA5 reanalysis data vectors to identify and screen extreme atmospheric/hydrodynamic forcing events (e.g., severe *Jugo* cyclonic sequences).
* 📄 **`generate_apisten_schematic.py`**: A dedicated diagnostic utility using the Graphviz engine to programmatically compile and output the formal vector architectural schematic layout of the network's processing pipelines.

---

## 📊 Data Availability & Reproducibility

To maintain absolute scientific reproducibility, this framework utilizes a dual-repository structure:
1. **Source Code Codebase (This Repository):** Contains the production-ready implementation scripts.
2. **Master Training Dataset & Multimedia (Zenodo Archive):** The full **4,000 footprint-representative stochastic realizations** used to bridge sub-pixel scale gaps, along with the 22 sequential test-polygon time-series arrays, are permanently archived on Zenodo at:  
🔗 **[https://doi.org/10.5281/zenodo.20688936](https://doi.org/10.5281/zenodo.20688936)**

### Data Setup Instructions:
1. Download `inputs.bin`, `outputs.bin`, and `XLpolygon_input_directory.zip` from the Zenodo repository.
2. Place all assets directly into your root execution folder.
3. Unzip the directory so that the 22 temporal binary files sit exactly under the path: `./XLpolygon_input_directory/XLpolygon_input_YYYYMMDD.bin`
4. Run `python3 APISTEN_Time_Series_Pipeline.py`

## ⚙️ Prerequisites & Setup

### 1. Python Environment
The code is built for **Python 3.8+** and requires standard scientific data packages. You can install the primary python requirements via `pip`:

```bash
pip3 install numpy pandas scipy scikit-learn graphviz
