# Documentation Index

**Thermodynamic Causal Inference for Cancer Drug Resistance**

Navigation hub for production-grade documentation.

---

## 🎯 Engineering Philosophy

### [ENGINEERING_PHILOSOPHY.md](ENGINEERING_PHILOSOPHY.md) ⭐
**Purpose**: Production device standards vs academic prototypes
**Core principles**:
- Physics-based thermodynamic inference (not statistical correlation)
- Hardware-aware design for TSU deployment
- Deterministic, bit-exact reproducibility
- Fail-fast validation, auditable provenance

**Read first** - Sets architectural philosophy for all implementation.

---

## 🚀 Quickstart

### [QUICKSTART.md](QUICKSTART.md)
**Purpose**: Production pipeline from data → validated predictions
**Contents**:
- 30-second setup
- Data acquisition guide
- Production pipeline (preprocessing complete ✅)
- Implementation roadmap (THRML model, inference, validation)
- Verification checklist

**Read second** - Immediate action items and current status.

### [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)
**Purpose**: Python environment configuration
**Contents**:
- Virtual environment setup
- THRML v0.1.3 installation
- GPU configuration (CUDA/Metal)
- Troubleshooting guide

---

## 📚 Core Methodology

### [RIGOROUS_METHODOLOGY.md](RIGOROUS_METHODOLOGY.md) ⭐
**Purpose**: Complete methodology documentation
**Sections**:
1. Data preprocessing (✅ implemented)
2. THRML model construction (🚧 in progress)
3. Causal inference via free energy
4. Physics-based validation
5. Pipeline execution

**Primary reference** - Read sections as you implement each component.

---

## 📊 Data Documentation

### [data/README.md](data/README.md)
**Purpose**: Data overview and current inventory
**Status**: Expression, methylation, Model.csv, GDSC IC50 data acquired

### [data/DATA_SOURCES.md](data/DATA_SOURCES.md)
**Purpose**: Data provenance, citations, licenses
**Contents**: DepMap/CCLE and GDSC attributions

### [data/VERSION_COMPATIBILITY_ANALYSIS.md](data/VERSION_COMPATIBILITY_ANALYSIS.md)
**Purpose**: How we align 2018 methylation with 2025 expression data
**Key insight**: Model.csv provides the mapping (98.7% overlap achieved)

### [data/DATA_INVENTORY.md](data/DATA_INVENTORY.md)
**Purpose**: Current data status and file sizes
**Use**: Verify downloads complete

### [data/READY_TO_PROCEED.md](data/READY_TO_PROCEED.md)
**Purpose**: Data acquisition completion checklist

---

## 🧬 Implementation Modules

### [core/DATA_LOADER_README.md](core/DATA_LOADER_README.md)
**Purpose**: Data preprocessing technical documentation
**Status**: ✅ Complete (production-grade, 820 lines)
**Contents**:
- CCLE/GDSC data loading
- IC50 stratification (p33/p67)
- EGFR pathway gene extraction (12 genes)
- Tertile discretization
- SHA-256 provenance tracking

### [core/DATA_LOADER_QUICK_REFERENCE.md](core/DATA_LOADER_QUICK_REFERENCE.md)
**Purpose**: Quick lookup for preprocessing functions

### [core/VALIDATION_README.md](core/VALIDATION_README.md)
**Purpose**: Physics-based validation specification
**Status**: 🚧 To implement
**Contents**:
- Detailed balance verification
- Ergodicity checks
- Free energy convergence
- IC50 validation metrics

### [docs/INFERENCE_API.md](docs/INFERENCE_API.md)
**Purpose**: Causal inference interface specification
**Status**: 🚧 To implement

### [docs/INFERENCE_QUICKSTART.md](docs/INFERENCE_QUICKSTART.md)
**Purpose**: Inference quick reference

---

## 🔬 THRML Reference

### [THRML_COMPREHENSIVE_DOCUMENTATION.md](THRML_COMPREHENSIVE_DOCUMENTATION.md)
**Purpose**: Complete THRML v0.1.3 reference
**Contents**:
- Hardware primitives (pbit, pdit, TSU architecture)
- Mathematical formalisms (EBMs, Gibbs sampling)
- Software API (Nodes, Blocks, Factors, Samplers)
- Usage patterns and examples

### [THRML_API_VERIFIED.md](THRML_API_VERIFIED.md) ⭐
**Purpose**: Verified API patterns against installed v0.1.3 package
**Critical for**: Implementing `core/thrml_model.py`
**Contents**:
- Correct signatures (CategoricalEBMFactor, sample_states, etc.)
- Working code patterns
- Key gotchas and fixes

---

## 📁 File Structure

```
thrml-cancer-decision-support/
├── README.md                              # Project overview
├── QUICKSTART.md                          # Start here
├── ENGINEERING_PHILOSOPHY.md              # Code standards
├── RIGOROUS_METHODOLOGY.md                # Complete methodology
├── ENVIRONMENT_SETUP.md                   # Setup guide
├── DOCUMENTATION_INDEX.md                 # This file
│
├── data/
│   ├── raw/ccle/                          # CCLE data
│   ├── raw/gdsc/                          # GDSC data
│   ├── processed/                         # Preprocessed .pkl files ✅
│   ├── README.md                          # Data overview
│   ├── DATA_SOURCES.md                    # Citations
│   ├── DATA_INVENTORY.md                  # File inventory
│   ├── VERSION_COMPATIBILITY_ANALYSIS.md  # 2018 vs 2025 alignment
│   └── READY_TO_PROCEED.md                # Completion checklist
│
├── core/
│   ├── data_loader.py                     # ✅ Production preprocessing
│   ├── DATA_LOADER_README.md              # Technical docs
│   ├── DATA_LOADER_QUICK_REFERENCE.md     # Quick lookup
│   ├── thrml_model.py                     # 🚧 THRML model
│   ├── inference.py                       # 🚧 Causal inference
│   ├── validation.py                      # 🚧 Validation
│   └── VALIDATION_README.md               # Validation spec
│
├── docs/
│   ├── INFERENCE_API.md                   # Inference interface
│   └── INFERENCE_QUICKSTART.md            # Inference guide
│
├── THRML_COMPREHENSIVE_DOCUMENTATION.md   # THRML deep dive
├── THRML_API_VERIFIED.md                  # Verified v0.1.3 patterns
│
└── scripts/
    ├── 01_download_data.sh                # Data download
    └── 02_run_inference.py                # 🚧 Main pipeline
```

---

## 🗺️ Navigation by Task

**I want to understand the project**:
→ Start: [README.md](README.md)
→ Then: [QUICKSTART.md](QUICKSTART.md)

**I want to set up the environment**:
→ [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)
→ Verify: `python3 -c "import thrml; print(thrml.__version__)"`

**I want to get data**:
→ [QUICKSTART.md](QUICKSTART.md) - Data Acquisition section
→ [data/DATA_INVENTORY.md](data/DATA_INVENTORY.md)

**I want to run preprocessing**:
→ [core/DATA_LOADER_README.md](core/DATA_LOADER_README.md)
→ Run: `python3 core/data_loader.py --help`

**I want to build the THRML model**:
→ [THRML_API_VERIFIED.md](THRML_API_VERIFIED.md) (verified patterns)
→ [RIGOROUS_METHODOLOGY.md](RIGOROUS_METHODOLOGY.md) § 2

**I want to implement inference**:
→ [docs/INFERENCE_API.md](docs/INFERENCE_API.md)
→ [RIGOROUS_METHODOLOGY.md](RIGOROUS_METHODOLOGY.md) § 3

**I want to validate predictions**:
→ [core/VALIDATION_README.md](core/VALIDATION_README.md)
→ [RIGOROUS_METHODOLOGY.md](RIGOROUS_METHODOLOGY.md) § 4

---

## ✅ Documentation Status

**Core Pillars** (4 files):
- ✅ README.md
- ✅ QUICKSTART.md
- ✅ ENGINEERING_PHILOSOPHY.md
- ✅ RIGOROUS_METHODOLOGY.md
- ✅ ENVIRONMENT_SETUP.md
- ✅ DOCUMENTATION_INDEX.md (this file)

**THRML Reference** (2 files):
- ✅ THRML_COMPREHENSIVE_DOCUMENTATION.md (verified accurate)
- ✅ THRML_API_VERIFIED.md (verified against v0.1.3)

**Module Documentation** (5 files):
- ✅ core/DATA_LOADER_README.md
- ✅ core/DATA_LOADER_QUICK_REFERENCE.md
- ✅ core/VALIDATION_README.md
- ✅ docs/INFERENCE_API.md
- ✅ docs/INFERENCE_QUICKSTART.md

**Data Documentation** (6 files):
- ✅ data/README.md
- ✅ data/DATA_SOURCES.md
- ✅ data/DATA_INVENTORY.md
- ✅ data/VERSION_COMPATIBILITY_ANALYSIS.md
- ✅ data/READY_TO_PROCEED.md
- ✅ data/processed/preprocessing_report.txt

**Total**: 23 production-quality documentation files

---

## 🔄 Last Updated

**Date**: 2025-11-16
**Status**: Documentation cleanup complete
**Changes**:
- Removed 20 development/build artifact files
- Consolidated 3 guides into QUICKSTART.md
- Verified THRML documentation against v0.1.3
- Streamlined navigation

---

**Ready to implement** → [QUICKSTART.md](QUICKSTART.md)

#### Section 1: Data Preprocessing Pipeline (Lines 1-400)
- Input data specifications
- Gene selection strategy (EGFR pathway)
- Cell line stratification (sensitive/resistant)
- Data alignment and filtering
- **Discretization to categorical states** (critical for THRML)
- Quality checks
- Final data format

**Read this when**: Implementing `core/data_loader.py`

#### Section 2: THRML Model Construction (Lines 401-600)
- Probabilistic graphical model design
- INDRA prior integration
- Factor weight initialization
- THRML factor construction (CategoricalEBMFactor)
- Block Gibbs sampling setup

**Read this when**: Implementing `core/thrml_model.py`

#### Section 3: Causal Inference Procedure (Lines 601-800)
- Model discrimination framework (ΔF method)
- Free energy computation
- Pairwise causal direction testing
- Network construction
- Network comparison (sensitive vs resistant)

**Read this when**: Implementing `core/inference.py`

#### Section 4: Validation Framework (Lines 801-950)
- Drug prediction from network changes
- IC50 validation against GDSC
- Statistical significance testing

**Read this when**: Implementing `core/validation.py`

#### Section 5: Demo Execution Workflow (Lines 951-1200)
- Complete pipeline script template
- Hour-by-hour timeline
- Expected outputs
- Contingency plans

**Read this when**: Implementing `scripts/02_run_inference.py`

**Code Templates**: Throughout (copy-paste ready)

---

## 📋 Executive Summary

### 6. [DATA_AND_METHODOLOGY_SUMMARY.md](DATA_AND_METHODOLOGY_SUMMARY.md)
**What it is**: High-level overview of entire approach
**Read this if**: You need to explain the project to someone else
**Reading time**: 30 minutes
**Key sections**:
- Data acquisition status
- Data flow diagram
- Mathematical rigor summary
- Validation metrics
- Success criteria

**Best for**: Presentations, project reports, stakeholder updates

---

## 📁 Supporting Documentation

### 7. [data/DATA_SOURCES.md](data/DATA_SOURCES.md)
**What it is**: Data provenance and citations
**Read this if**: You need to cite data sources or understand licensing
**Key info**:
- CCLE/DepMap citations
- GDSC citations
- Data license terms
- Update frequency

### 8. [THRML_COMPREHENSIVE_DOCUMENTATION.md](THRML_COMPREHENSIVE_DOCUMENTATION.md)
**What it is**: Complete THRML API reference (from Extropic)
**Read this if**: You need to understand THRML primitives and API
**Key sections**:
- Hardware primitives (pbit, pdit, pmode, pmog)
- Software abstractions (Nodes, Blocks, Factors)
- Mathematical formalisms
- Usage patterns

**Note**: This is reference material, not specific to this project

### 9. [TECHNICAL_ASSESSMENT.md](TECHNICAL_ASSESSMENT.md)
**What it is**: Technical challenges and solutions
**Read this if**: You encounter implementation issues

### 10. [DELIVERY_SUMMARY.md](DELIVERY_SUMMARY.md)
**What it is**: Project deliverables checklist

### 11. [PACKAGE_MANIFEST.md](PACKAGE_MANIFEST.md)
**What it is**: Python dependencies and versions

---

## 🎯 Quick Navigation by Task

### I want to understand the project
→ Start: [README.md](README.md)
→ Then: [DATA_AND_METHODOLOGY_SUMMARY.md](DATA_AND_METHODOLOGY_SUMMARY.md)

### I want to get data
→ Start: [QUICK_START.md](QUICK_START.md) (Critical section)
→ If issues: [DATA_DOWNLOAD_SUMMARY.md](data/DATA_DOWNLOAD_SUMMARY.md)
→ Run: [scripts/01_download_data.sh](scripts/01_download_data.sh)

### I want to implement preprocessing
→ Read: [RIGOROUS_METHODOLOGY.md § 1](RIGOROUS_METHODOLOGY.md#1-data-preprocessing-pipeline)
→ Implement: `core/data_loader.py`
→ Template code: In methodology doc

### I want to build the THRML model
→ Read: [RIGOROUS_METHODOLOGY.md § 2](RIGOROUS_METHODOLOGY.md#2-thrml-model-construction)
→ Reference: [THRML_COMPREHENSIVE_DOCUMENTATION.md](THRML_COMPREHENSIVE_DOCUMENTATION.md)
→ Implement: `core/thrml_model.py`

### I want to do causal inference
→ Read: [RIGOROUS_METHODOLOGY.md § 3](RIGOROUS_METHODOLOGY.md#3-causal-inference-procedure)
→ Implement: `core/inference.py`

### I want to validate predictions
→ Read: [RIGOROUS_METHODOLOGY.md § 4](RIGOROUS_METHODOLOGY.md#4-validation-framework)
→ Implement: `core/validation.py`

### I want to run the full pipeline
→ Read: [RIGOROUS_METHODOLOGY.md § 5.1](RIGOROUS_METHODOLOGY.md#51-complete-pipeline-script)
→ Implement: `scripts/02_run_inference.py`
→ Timeline: [RIGOROUS_METHODOLOGY.md § 5.2](RIGOROUS_METHODOLOGY.md#52-demo-execution-timeline)

### I want to present results
→ Outputs: [RIGOROUS_METHODOLOGY.md § 5.3](RIGOROUS_METHODOLOGY.md#53-expected-outputs)
→ Metrics: [DATA_AND_METHODOLOGY_SUMMARY.md § Part 5](DATA_AND_METHODOLOGY_SUMMARY.md#part-5-validation-metrics)

---

## 📖 Recommended Reading Order

### For Implementation (8-12 hours)
1. [QUICK_START.md](QUICK_START.md) - 10 min
2. Download CCLE data manually - 30 min
3. [RIGOROUS_METHODOLOGY.md § 1](RIGOROUS_METHODOLOGY.md) - 30 min, implement `data_loader.py` - 2 hrs
4. [RIGOROUS_METHODOLOGY.md § 2](RIGOROUS_METHODOLOGY.md) - 20 min, implement `thrml_model.py` - 1 hr
5. [RIGOROUS_METHODOLOGY.md § 3](RIGOROUS_METHODOLOGY.md) - 30 min, implement `inference.py` - 2 hrs
6. [RIGOROUS_METHODOLOGY.md § 4](RIGOROUS_METHODOLOGY.md) - 20 min, implement `validation.py` - 1 hr
7. [RIGOROUS_METHODOLOGY.md § 5](RIGOROUS_METHODOLOGY.md) - 30 min, implement `02_run_inference.py` - 1 hr
8. Test and run - 2-4 hrs

### For Understanding (2 hours)
1. [README.md](README.md) - 5 min
2. [QUICK_START.md](QUICK_START.md) - 10 min
3. [DATA_AND_METHODOLOGY_SUMMARY.md](DATA_AND_METHODOLOGY_SUMMARY.md) - 30 min
4. [RIGOROUS_METHODOLOGY.md](RIGOROUS_METHODOLOGY.md) (skim all sections) - 1 hr

### For Troubleshooting
1. Check [QUICK_START.md § Fallback Plans](QUICK_START.md#fallback-plans)
2. Check [DATA_DOWNLOAD_SUMMARY.md § Troubleshooting](data/DATA_DOWNLOAD_SUMMARY.md#troubleshooting)
3. Check [RIGOROUS_METHODOLOGY.md § 5.4](RIGOROUS_METHODOLOGY.md#54-contingency-plans)

---

## 🗂️ File Structure Map

```
thrml-cancer-decision-support/
│
├─ DOCUMENTATION_INDEX.md              ← YOU ARE HERE
│
├─ README.md                            ← Start here (project overview)
├─ QUICK_START.md                       ← Next steps (action items)
├─ RIGOROUS_METHODOLOGY.md              ← ⭐ Core methodology (1200 lines)
├─ DATA_AND_METHODOLOGY_SUMMARY.md      ← Executive summary
│
├─ THRML_COMPREHENSIVE_DOCUMENTATION.md ← THRML API reference
├─ TECHNICAL_ASSESSMENT.md              ← Technical challenges
├─ DELIVERY_SUMMARY.md                  ← Deliverables checklist
├─ PACKAGE_MANIFEST.md                  ← Dependencies
│
├─ data/
│  ├─ DATA_SOURCES.md                   ← Citations and provenance
│  ├─ DATA_DOWNLOAD_SUMMARY.md          ← Download guide
│  ├─ raw/                              ← Downloaded datasets
│  └─ processed/                        ← Preprocessed outputs
│
├─ scripts/
│  ├─ 01_download_data.sh               ← Automated download
│  ├─ 02_run_inference.py               ← Main pipeline (to implement)
│  └─ 03_analyze_results.py             ← Analysis (to implement)
│
├─ core/
│  ├─ data_loader.py                    ← Preprocessing (to implement)
│  ├─ thrml_model.py                    ← THRML model (to implement)
│  ├─ indra_client.py                   ← INDRA API (partial)
│  ├─ inference.py                      ← Causal inference (to implement)
│  └─ validation.py                     ← IC50 validation (to implement)
│
└─ results/                             ← Output directory
```

---

## 🔗 Cross-References

### Methodology → Code
- [RIGOROUS_METHODOLOGY.md § 1.5](RIGOROUS_METHODOLOGY.md#15-discretization-to-categorical-states) → Implement in `core/data_loader.py::discretize_tertiles()`
- [RIGOROUS_METHODOLOGY.md § 2.2](RIGOROUS_METHODOLOGY.md#22-indra-prior-integration) → Implement in `core/indra_client.py::query_indra_interactions()`
- [RIGOROUS_METHODOLOGY.md § 3.2](RIGOROUS_METHODOLOGY.md#32-free-energy-computation) → Implement in `core/inference.py::estimate_free_energy()`
- [RIGOROUS_METHODOLOGY.md § 4.2](RIGOROUS_METHODOLOGY.md#42-validation-against-gdsc-ic50-data) → Implement in `core/validation.py::validate_drug_predictions()`

### Data Sources → Usage
- [data/DATA_SOURCES.md](data/DATA_SOURCES.md) → Referenced in `core/data_loader.py` for file paths
- [DATA_DOWNLOAD_SUMMARY.md](data/DATA_DOWNLOAD_SUMMARY.md) → URLs used in `scripts/01_download_data.sh`

### Dependencies → Installation
- [PACKAGE_MANIFEST.md](PACKAGE_MANIFEST.md) → Install with `requirements.txt`
- [QUICK_START.md § Critical Dependencies](QUICK_START.md#critical-dependencies) → Setup commands

---

## 📊 Documentation Status

| Document | Status | Purpose | Priority |
|----------|--------|---------|----------|
| DOCUMENTATION_INDEX.md | ✓ Complete | Navigation hub | High |
| README.md | ✓ Complete | Project overview | High |
| QUICK_START.md | ✓ Complete | Action items | Critical |
| RIGOROUS_METHODOLOGY.md | ✓ Complete | Core methodology | Critical |
| DATA_AND_METHODOLOGY_SUMMARY.md | ✓ Complete | Executive summary | Medium |
| DATA_DOWNLOAD_SUMMARY.md | ✓ Complete | Data acquisition | High |
| data/DATA_SOURCES.md | ✓ Complete | Citations | Medium |
| THRML_COMPREHENSIVE_DOCUMENTATION.md | ✓ Complete | THRML reference | Medium |
| scripts/01_download_data.sh | ✓ Complete | Download automation | High |
| scripts/02_run_inference.py | ⚠️ Template only | Main pipeline | Critical |
| scripts/03_analyze_results.py | ❌ Not created | Analysis | High |
| core/data_loader.py | ⚠️ Partial | Preprocessing | Critical |
| core/thrml_model.py | ⚠️ Partial | THRML model | Critical |
| core/inference.py | ❌ Not created | Causal inference | Critical |
| core/validation.py | ❌ Not created | IC50 validation | Critical |

**Legend**:
- ✓ Complete: Ready to use
- ⚠️ Partial: Needs implementation
- ❌ Not created: Needs full implementation

---

## 🎓 Learning Path

### Level 1: Understanding (30 min)
Read these to understand what the project does:
1. [README.md](README.md)
2. [QUICK_START.md § 30-Second Overview](QUICK_START.md#30-second-overview)
3. [DATA_AND_METHODOLOGY_SUMMARY.md § Part 3](DATA_AND_METHODOLOGY_SUMMARY.md#part-3-data-usage-in-hackathon-demo)

### Level 2: Data Acquisition (1 hour)
Read these to get the data:
1. [QUICK_START.md § Critical](QUICK_START.md#-critical-manual-data-download-required)
2. [DATA_DOWNLOAD_SUMMARY.md](data/DATA_DOWNLOAD_SUMMARY.md)
3. Run [scripts/01_download_data.sh](scripts/01_download_data.sh)

### Level 3: Implementation Prep (2 hours)
Read these before coding:
1. [RIGOROUS_METHODOLOGY.md § 1-2](RIGOROUS_METHODOLOGY.md)
2. [THRML_COMPREHENSIVE_DOCUMENTATION.md § Part III](THRML_COMPREHENSIVE_DOCUMENTATION.md)
3. [QUICK_START.md § Key Technical Decisions](QUICK_START.md#key-technical-decisions-already-made)

### Level 4: Full Implementation (8-12 hours)
Follow this sequence:
1. Implement following [RIGOROUS_METHODOLOGY.md § 1-5](RIGOROUS_METHODOLOGY.md)
2. Test with quick mode
3. Run full inference
4. Validate and analyze

---

## 🆘 Help & Troubleshooting

### Common Issues

**"I can't download CCLE data"**
→ See: [DATA_DOWNLOAD_SUMMARY.md § Troubleshooting](data/DATA_DOWNLOAD_SUMMARY.md#troubleshooting)
→ Alternative: Use Figshare links in that document

**"THRML import fails"**
→ See: [QUICK_START.md § Critical Dependencies](QUICK_START.md#critical-dependencies)
→ Run: `python scripts/00_test_environment.py`

**"I don't understand the methodology"**
→ Start with: [DATA_AND_METHODOLOGY_SUMMARY.md § Part 4](DATA_AND_METHODOLOGY_SUMMARY.md#part-4-mathematical-rigor)
→ Then: [RIGOROUS_METHODOLOGY.md](RIGOROUS_METHODOLOGY.md) with section-by-section reading

**"The code doesn't work"**
→ Check: [RIGOROUS_METHODOLOGY.md § 5.4 Contingency Plans](RIGOROUS_METHODOLOGY.md#54-contingency-plans)
→ Fallbacks are provided for all critical components

**"I'm running out of time"**
→ See: [QUICK_START.md § Fallback Plans](QUICK_START.md#fallback-plans)
→ Quick test mode: `--quick-test --synthetic-data`

---

## 📝 Document Conventions

### Notation
- **Bold**: Critical information, action items
- *Italic*: Variable names, technical terms
- `Code`: File paths, function names, commands
- → : Navigation link (read this next)
- ✓ : Completed/available
- ⚠️ : Needs attention/partial
- ❌ : Not yet created

### Section References
- § = Section (e.g., "§ 1.5" = Section 1.5)
- Lines X-Y = Line numbers in document

### File Paths
- Relative to project root: `data/raw/ccle/file.csv`
- Absolute paths shown when needed

---

## 🔄 Last Updated
**Date**: November 16, 2024
**By**: Claude Code
**Version**: 1.0

**Change Log**:
- Initial creation of documentation index
- All core methodology documents complete
- Data download infrastructure ready
- Implementation templates provided

---

## 📞 Contact

For questions about:
- **Data sources**: See citations in [data/DATA_SOURCES.md](data/DATA_SOURCES.md)
- **THRML API**: Refer to [THRML_COMPREHENSIVE_DOCUMENTATION.md](THRML_COMPREHENSIVE_DOCUMENTATION.md)
- **Methodology**: All details in [RIGOROUS_METHODOLOGY.md](RIGOROUS_METHODOLOGY.md)

---

**Ready to start?** → [QUICK_START.md](QUICK_START.md)
