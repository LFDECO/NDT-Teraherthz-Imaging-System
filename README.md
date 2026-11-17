# THz-AI Polymer NDT & Additive Manufacturing Diagnostics

**Collaborative Research Initiative:** SRM University & CSIR-Central Scientific Instruments Organisation (CSIR-CSIO) Terahertz Imaging and Additive Manufacturing research collaboration.

---

## 1. Folder Responsibility & Scope

This repository implements the standalone **AI Diagnostic & Subsurface Defect Segmentation Pipeline** for Non-Destructive Testing (NDT) of Fused Deposition Modeling (FDM) 3D-printed polymer parts scanned via Terahertz (THz) reflection imaging ($0.135\text{–}0.450\text{ THz}$).

In alignment with the **October 13, 2026 CSIR-CSIO Project Scoping Meeting**, the deliverable scope of this repository is strictly the **standalone predictive models, signal conditioning, quantitative metrology, and verification suite**—with **no hardware integration** into physical 3D printers or THz raster stage drivers.

### Core Diagnostic Capabilities (4 Tasks, 5 Outputs)
1. **Model 1 — Global Structural Characterizer (`ConvNeXt` / `ResNet-34` / Physics-Feature Heads):**
   - **Material Classification:** `PLA`, `PETG`, `ABS`, `TPU`
   - **Infill Pattern Classification:** `Rectilinear`, `Grid`, `Gyroid`, `Concentric` (pending multi-pattern scans from partner AM lab)
   - **Infill Density Regression:** Continuous percentage ($10\%\text{–}100\%$)
2. **Model 2 — Subsurface Defect Segmentor (`THz-Adapted DeepLabV3+`):**
   - Pixel-level semantic segmentation of **blind defects** (surface-open calibration voids) and **buried defects** ($100\%$ encapsulated internal voids, Layer-3 power-interruption cold-joint delaminations, and filament bulges)
3. **Layer 5 — Quantitative Metrology (OpenCV):**
   - Extracts physical centroid $(X, Y)\text{ mm}$, apparent area ($\text{mm}^2$), apparent diameter ($\text{mm}$), and beam-blur-deconvolved equivalent diameter ($d_{\text{true}} \approx \sqrt{\max(0.5^2, d_{\text{app}}^2 - d_{\text{beam}}^2)}$) into standardized JSON inspection reports.

### Repository Directory Map
```text
NDT Teraherthz Imaging System/
├── README.md                              # Master repository architecture & agent governance
├── requirements.txt                       # Reproducible Python dependencies
├── Polymer sample data/                   # 15 real THz CSV matrices + MATLAB PNG visualizations
├── docs/                                  # (Local/Private via .gitignore) PRD & architecture specs
├── configs/                               # Centralized YAML hyperparameters (RGB demo & THz NDT)
├── data/                                  # Generated metadata manifest & ground-truth/reference masks
│   └── masks/                             # Reference defect segmentation masks (.npy / .png)
├── src/                                   # Modular Python package
│   ├── preprocessing/                     # CSV loader, hot-strip masking, Wiener/DoG norm, augmentations
│   ├── models/                            # Model 1 Characterizer, DeepLabV3+ (RGB & THz), U-Net baseline, losses
│   ├── metrology/                         # OpenCV spatial moment metrology & JSON report formatter
│   └── visualization/                     # CSIR presentation figure generators
├── scripts/                               # End-to-end runnable verification & CSIR demo scripts
├── outputs/                               # Generated figures and JSON diagnostic reports
│   ├── figures/                           # CSIR meeting slides/figures (01 to 06)
│   └── reports/                           # JSON inspection reports & benchmark metrics
├── docker/                                # Containerization for SRM College GPU Lab training
└── tests/                                 # Automated verification & governance compliance suite
```

---

## 2. Step-by-Step Workflow & Verification Protocol (For Agents & Engineers)

Every engineer or AI agent onboarding to this repository must execute the verification pipeline sequentially:

```powershell
# 1. Install dependencies (if not already present)
pip install -r requirements.txt

# 2. Verify all 15 CSV scans, generate data/metadata_manifest.csv & masks, and plot preprocessing diagnostics
python scripts/verify_dataset_and_preprocess.py

# 3. Run the CSIR Meeting DeepLabV3+ Demo (RGB Architecture Explainer + THz Adaptation + Real THz Defect Localization)
python scripts/run_csir_deeplab_demo.py

# 4. Run Model 1 Material & Infill Separability Audit across Cohort B scans
python scripts/verify_model1_separability.py

# 5. Execute the full automated test & governance verification suite
python tests/test_pipeline_verification.py
```

### Verification Criteria
- All 15 CSV scans in `Polymer sample data/` parse cleanly as 2D floating-point matrices without `NaN`/`Inf`.
- `DeepLabV3Plus` runs cleanly in both **RGB Demo Mode** (`in_channels=3, aspp_rates=(6, 12, 18), output_stride=16`) and **THz NDT Mode** (`in_channels=3, upsample_size=(128, 128), output_stride=8, aspp_rates=(1, 2, 4)`).
- Every folder in the repository contains a `README.md` with the mandatory 5-section governance template and an up-to-date Changelog.

---

## 3. Architectural Rationale: Why This Approach Over All Alternatives

1. **Decoupled Two-Model Design vs. Single Monolithic Multi-Task Network:**
   - Training a single shared backbone for both bulk macro-properties (material, pattern, infill %) and fine pixel-level defect masks causes severe gradient conflict on small datasets ($15\text{–}60$ scans). Decoupling lets Model 1 optimize global texture/attenuation statistics while Model 2 (`DeepLabV3+`) optimizes localized spatial contrast and edge boundaries.
2. **THz-Adapted `DeepLabV3+` (Stride-8, Upsampled $128\times 128$, ASPP $r\in\{1,2,4\}$) vs. Naive Stride-16 `DeepLabV3+` (`ResNet-101`):**
   - Standard `DeepLabV3+` uses output stride 16 and ASPP rates $(6, 12, 18)$ designed for $512\times 512$ RGB scenes. Applying stride 16 directly to native $38\times 38$ or $46\times 46$ THz scans collapses bottleneck feature maps to $2\times 2$ or $3\times 3$ pixels, where $r=12$ and $r=18$ sample purely outside the image (zero-padding). Upsampling THz inputs to $128\times 128$, using a lightweight `ResNet-18` stride-8 encoder ($16\times 16$ bottleneck), and scaling ASPP dilations to $r\in\{1,2,4\}$ preserves physical defect boundaries and $3\text{–}4\text{ mm}$ beam-waist context.
3. **`DeepLabV3+` vs. `YOLOv8-seg`, `PatchCore`, `U-Net`, and Vision Transformers (`SegFormer`/`Swin`):**
   - `YOLOv8-seg` relies on bounding-box proposals that break down on diffuse, diffraction-blurred subsurface delamination seams.
   - `PatchCore` produces uncalibrated anomaly heatmaps without crisp closed contours or defect-type separation (kept strictly as a comparison idea).
   - Standard `U-Net` lacks parallel multi-scale dilated ASPP kernels to separate periodic infill patterns from aperiodic buried voids, though a lightweight `SmallUNet` is retained in `src/models/model2_unet_baseline.py` as an ablation baseline.
   - Heavy ViTs (`SegFormer-B5`, `Swin`) lack translation equivariance and overfit severely under $N < 50$ physical THz scans.

---

## 4. What Could Be Done Better & Known Constraints

- **Dataset Size & Labels:** Only 15 real CSV scans exist currently (Cohort A: 5 PLA $100\%$ infill defect-geometry scans; Cohort B: 10 multi-material/multi-infill plate scans). Pixel-wise ground-truth CAD/caliper defect masks have not yet been supplied by CSIR-CSIO; current masks in `data/masks/` are physics-derived reference masks extracted from intensity anomalies and named geometry (`1C`, `1S`, `3C`, `3S`), which introduces optimistic bias until verified ground-truth masks arrive.
- **Acquisition Confounders Observed in Real Data:** Cohort B varies frequency alongside material/infill (`0.375`, `0.4`, `0.425`, `0.45 THz`), has signed baseline-subtracted values (down to $-0.116$), and exhibits border hot-strip artifacts in columns `0..3` and `57..60`. Standardized scanner gain, empty-holder background scans, and fixed frequency protocols are required for Model 1 production training.
- **Pattern & Buried Defect Validation:** All current scans use one default printing pattern; multi-pattern (`Grid`, `Gyroid`, `Concentric`) and confirmed Layer-3 buried power-interruption defects are pending from CSIR-CSIO.

---

## 5. Mandatory Agent Update & Changelog Governance Policy

> [!IMPORTANT]
> **AGENT GOVERNANCE RULE (NON-NEGOTIABLE):**
> Every AI agent or human contributor who modifies, adds, deletes, or refactors any file in ANY folder of this repository **MUST** update the `README.md` inside **that specific folder** (and this root `README.md` if repository-wide architecture or dependencies change). Every Changelog update **MUST** explicitly document all five fields:
> 1. **What changes were made** (exact files, symbols, hyperparameters, or scripts modified)
> 2. **Why these changes were made** (data evidence, error observed, or meeting requirement)
> 3. **What impact these changes had** (verified metrics, figure outputs, or behavior changes)
> 4. **What could have been done better** (remaining limitations, assumptions, or technical debt)
> 5. **Why this approach was considered over all other available approaches** (comparative trade-off)

### Repository Changelog

| Date | Author / Agent | What Changed | Why Changed | Impact of Change | What Could Be Done Better | Why Chosen Over Alternatives |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-11-17 | Antigravity Execution Agent | Created complete repository structure (`Polymer sample data`, `docs`, `configs`, `data`, `src`, `scripts`, `outputs`, `docker`, `tests`), PRD, DeepLabV3+ (RGB + THz), Model 1 scaffold, OpenCV metrology, and folder governance `README.md` files. | Reconcile project context, Oct 13 CSIR meeting scope, and the 15 real THz CSV scans into an actionable, verifiable engineering baseline ahead of the next CSIR review. | Established reproducible end-to-end verification, RGB + THz `DeepLabV3+` visualizers, quantitative metrology JSON outputs, and strict multi-agent changelog traceability across all folders. | Obtain verified CAD/G-code ground-truth defect masks, complex phase/time-domain waveforms, and multi-pattern scans from CSIR-CSIO to replace physics-derived reference masks. | Selected decoupled `ResNet` + THz-adapted `DeepLabV3+` over monolithic networks, `YOLOv8-seg`, or heavy ViTs due to small-sample stability, $3\text{–}4\text{ mm}$ beam diffraction blur, and millimeter metrology requirements. |
| 2026-11-17 | Antigravity Execution Agent | Added `.gitignore`, `.env.example`, and CUDA/CPU auto-dispatch in `scripts/run_csir_deeplab_demo.py` and `scripts/verify_model1_separability.py`. | Protect credentials/checkpoints/optional NDA scans before GitHub push and enable automatic GPU execution when CUDA PyTorch is present. | Prevents accidental secret commits and allows zero-edit switching between CPU laptops and CUDA GPUs/Docker. | Configure DVC with a private Google Drive remote once CSIR scan counts grow beyond 40+ samples. | `.gitignore` + `.env.example` + private Drive/DVC separation protects sensitive artifacts better than committing secrets or bloated binary weights to Git history. |
| 2026-11-17 | Antigravity Execution Agent | Updated `.gitignore` to exclude all `.md` files except root `README.md` (`*.md` + `!/README.md`) while keeping `Polymer sample data/` tracked. | Keep internal PRDs, meeting briefs, and folder governance docs private on public GitHub pushes while sharing code and current sample scans openly. | Only root `README.md` is exposed publicly alongside code, configs, data, and outputs; checkpoints and internal docs remain local/private. | Use `git-crypt` or a private companion repo (`docs` + `checkpoints`) so collaborators sync ignored `.md` files automatically without Google Drive. | Pattern-level `*.md` + `!/README.md` prevents accidental leaks of internal research notes on public pushes without hiding project entry documentation. |
| 2026-11-17 | Antigravity Execution Agent | Removed personal/mentor names from public `README.md`, `scripts/verify_model1_separability.py`, and regenerated `outputs/reports/model1_separability_summary.json`. | Ensure no personal names are exposed in any publicly tracked files before pushing via GitHub Desktop. | Zero personal names remain in any tracked repository file while preserving technical attribution to SRM University and CSIR-CSIO. | Automate PII/name regex checks in CI pre-commit hooks before public pushes. | Anonymizing tracked files prevents unintended exposure on public GitHub while keeping scientific provenance clear. |
