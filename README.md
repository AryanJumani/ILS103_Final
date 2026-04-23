# Determining Performance Thresholds for AI-Enhanced 3D Mobile Rendering

**Author:** Aryan Naresh Jumani  
**Research Question:** To what extent can AI-based features be integrated into 3D mobile environments before graphics latency exceeds acceptable limits?  
**License (Data):** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
**License (Code):** [MIT](LICENSE.md)  
**Archived DOI:** *(to be assigned upon Zenodo deposit)*

---

## Repository Structure

```
.
├── README.md
├── LICENSE.md
├── final_data/
│   └── merged_metrics.csv          # Final analytical dataset (output of pipeline)
├── analysis_scripts/
│   ├── model.py                    # Regression and threshold modelling
│   ├── viz.py                      # Visualization generation
│   └── requirements.txt            # Python dependencies
├── raw_data/
│   ├── Antutu/
│   │   ├── Android_SoC.csv         # SoC GPU/CPU/Total benchmark scores
│   │   ├── Android_AI_General.csv  # Per-task AI inference scores
│   │   ├── Android_AI_LLM.csv      # LLM inference scores (not used in model)
│   │   ├── Android_Performance_Smartphone.csv   # (not used in model)
│   │   ├── Android_Performance_Pad.csv          # (not used in model)
│   │   └── iOS_Performance.csv                  # (not used in model)
│   └─ DeepEn/
│       ├── Kernel_latency/         # Per-kernel latency JSON files (ms)
│       ├── Kernel_energy/          # Per-kernel energy JSON files (mJ)
│       ├── Kernel_power/           # Per-kernel power JSON files (mW)
│       └── Predictor/              # Pre-trained latency predictor models (.pkl)
└── visualization/
    └── viz.png                     # Output chart from viz.py
```

---

## Final Dataset: `final_data/merged_metrics.csv`

This is the primary analytical output of the data pipeline. It contains **44 rows** and **7 columns**, covering **9 distinct Snapdragon SoC tiers** across a range of Android devices. There are no missing values in any column.

### Column Definitions

| Column | Type | Units | Source | Description |
|--------|------|-------|--------|-------------|
| `Device_Name` | string | — | `Android_AI_General.csv` | Commercial device model name (e.g., *nubia Z80 Ultra*) |
| `Normalized_SoC` | string | — | Derived | Lowercase, standardized SoC name used as the primary join key across datasets (e.g., `snapdragon 8 elite gen 5`). Raw SoC name strings from Antutu were cleaned to remove clock speed annotations and ranking prefixes before merging. |
| `GPU Score` | float | Antutu points | `Android_SoC.csv` | Raw Antutu GPU subscore for the device's SoC. Range: 106,936 – 1,352,477. |
| `Antutu_AI_Score` | float | Antutu points | `Android_AI_General.csv` | Total composite AI inference score summing image classification, object detection, super-resolution, and style transfer subtasks. Range: 123,726 – 2,609,285. |
| `Device_AI_Kernel_Latency_ms` | float | milliseconds | `DeepEn/Kernel_energy/conv-bn-relu_energy.json` | Estimated per-frame AI kernel latency for the Conv-BN-ReLU operation, the most common building block in mobile neural networks. Derived from DeepEn's device-level measurements using the anonymous device-ID-to-SoC mapping documented below. Range: 5.668 – 119.524 ms. |
| `GPU_Percentile` | float | 0.0 – 1.0 | Derived from `GPU Score` | Percentile rank of the device's GPU score within the merged dataset. 1.0 = highest GPU score in the sample. Calculated using `pandas.Series.rank(pct=True)`. |
| `Theoretical_Max_AI_Ops_Per_Frame` | int | ops/frame (0–2) | Derived | Binned threshold metric representing the estimated maximum number of simultaneous AI operations a device can sustain per render frame without exceeding latency limits. Values: 0 (below threshold), 1 (marginal), 2 (capable). Derived from `GPU_Percentile` and `Device_AI_Kernel_Latency_ms` via the regression model in `model.py`. |

### SoC Tiers Represented

All 9 SoCs in the dataset are Qualcomm Snapdragon:

```
snapdragon 870
snapdragon 7 gen 1
snapdragon 7+ gen 2
snapdragon 8 gen 2
snapdragon 8 gen 3
snapdragon 8s gen 3
snapdragon 8s gen 4
snapdragon 8 elite
snapdragon 8 elite gen 5
```

---

## Raw Data Sources

### 1. Antutu — `Android_SoC.csv`
**Used in pipeline: yes**  
Provides ranked SoC benchmark scores for Android chips. Key columns used: `GPU Score`, `Device` (SoC name). The `Device` field includes ranking prefixes and CPU clock annotations that were stripped during normalization (see Data Cleaning below).

### 2. Antutu — `Android_AI_General.csv`
**Used in pipeline: yes**  
Provides per-device AI inference benchmark scores broken down by task type: image classification, object detection, super-resolution, and style transfer. The `Total Score` column was used as `Antutu_AI_Score`. Device names from this file served as the base for `Device_Name` in the merged dataset.

### 3. DeepEn2023 — `Kernel_energy/conv-bn-relu_energy.json`
**Used in pipeline: yes**  
Provides per-kernel energy, power, and latency measurements collected directly on mobile hardware. Each JSON file is structured as:

```json
{
  "conv-bn-relu": {
    "DEVICE_ID": { "energy": "..." },
    ...
  }
}
```

**Important:** The keys (e.g., `GTFT8Y`, `HH76RD`) are anonymous hardware identifiers, not SoC model names. These were resolved to SoC names using the device-ID mapping table in `DeepEn/Predictor/predictors.yaml` and the `kernel_config.zip` configuration files. The resolved mapping is preserved in `final_data/merged_metrics.csv` via the `Normalized_SoC` column. The `conv-bn-relu` kernel was selected because it is the most computationally dominant layer type in the MobileNet-family architectures commonly deployed for on-device AI rendering tasks.

### 4. Smartphone Processors Ranking — `ML_ALL_benchmarks.csv`, `smartphone_cpu_stats.csv`, `antutu_android_vs_ios_v4.csv`
**Used in pipeline: no (supporting context only)**  
Provides cross-platform CPU, GPU, and NPU benchmark scores including iOS devices. These files were examined during exploratory analysis to validate score ranges but were not merged into the final dataset due to inconsistent SoC naming conventions with the Antutu sources and limited overlap with the Android-only scope of the final model.

---

## Data Cleaning and Transformation

All cleaning and merging logic is implemented in `model.py`. The full steps, in order, are:

**Step 1 — SoC name normalization.**
The `Device` field in `Android_SoC.csv` contains entries like:
```
1\n                                    Qualcomm Snapdragon 8 Elite Gen 5\n                                    (2x 4.6GHz ...)
```
Ranking prefixes, newlines, clock speed annotations, and manufacturer prefixes (`Qualcomm`) were stripped using `str.strip()`, regex substitution, and lowercasing to produce the `Normalized_SoC` key (e.g., `snapdragon 8 elite gen 5`).

**Step 2 — DeepEn device-ID resolution.**
Anonymous device identifiers in the DeepEn JSON files were mapped to SoC names using the predictor configuration files. Only device IDs with a confirmed SoC match in the Antutu dataset were retained; unresolvable IDs were dropped.

**Step 3 — Dataset merge.**
`Android_AI_General.csv` and `Android_SoC.csv` were merged on `Normalized_SoC`. DeepEn kernel measurements were then joined on the same key. Rows with no matching SoC across all three sources were excluded from the final dataset.

**Step 4 — GPU Percentile calculation.**
`GPU_Percentile` was computed using `pandas.Series.rank(pct=True)` on the `GPU Score` column of the merged dataset.

**Step 5 — Threshold derivation.**
`Theoretical_Max_AI_Ops_Per_Frame` was computed by the regression model in `model.py` using `GPU_Percentile` and `Device_AI_Kernel_Latency_ms` as inputs. The binning thresholds are defined and documented within `model.py`.

---

## Reproducing the Analysis

**Requirements:** Python 3.9+. Install dependencies with:

```bash
pip install -r analysis_scripts/requirements.txt
source venv/bin/activate
```

**Run the full pipeline from raw data:**

```bash
python analysis_scripts/model.py
python analysis_scripts/viz.py
```

`model.py` reads from `raw_data/` and writes `final_data/merged_metrics.csv`.  
`viz.py` reads `final_data/merged_metrics.csv` and writes `visualization/viz.png`.

Git commits were made at each major processing milestone (raw ingest, normalization, merge, final output) to maintain a transparent audit trail. All commits are visible in the repository history.

---

## Licensing

| Asset | License |
|-------|---------|
| `final_data/merged_metrics.csv` and all processed outputs | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| All `.py` and `.ipynb` scripts | [MIT](LICENSE.md) |

Downstream users may freely use, adapt, and redistribute any content in this repository provided appropriate credit is given to this project and to the original dataset sources listed above.

---

## Citation

If you use this dataset or code, please cite:

> Jumani, A. N. (2025). *Determining Performance Thresholds for AI-Enhanced 3D Mobile Rendering* [Dataset and analysis code]. GitHub. *(DOI to be added upon Zenodo deposit)*

---

## Original Data Sources

- Antutu Benchmark: https://www.kaggle.com/datasets/ireddragonicy/antutu-benchmark?resource=download
- DeepEn2023: https://amai-gsu.github.io/DeepEn2023/
