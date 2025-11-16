# Documentation Index

**XTR-0 Hackathon: Thermodynamic Causal Inference for Drug Response Prediction**

This is your navigation hub for all project documentation. Documents are organized by purpose and reading order.

---

## 🎯 Engineering Philosophy (READ FIRST)

### 0. [ENGINEERING_PHILOSOPHY.md](ENGINEERING_PHILOSOPHY.md) ⭐ **FOUNDATIONAL**
**What it is**: Production device engineering philosophy vs academic prototype
**Read this if**: You're about to write any code for this project
**Reading time**: 20 minutes
**Core principles**:
- **Physics, not statistics**: Thermodynamic truth vs p-values
- **Hardware-aware**: TSU simulation → future ASIC deployment
- **Deterministic**: Bit-exact reproducibility for medical devices
- **Fail-fast**: Invalid states impossible, not just unlikely
- **Auditable**: Clinical-grade provenance and error handling

**Why read this first**: Sets architectural philosophy that governs all implementation decisions. Distinguishes this from bioinformatics cruft.

**Next**: → [QUICK_START.md](#quick_startmd) to apply these principles

---

## 🚀 Start Here (New to This Project)

### 1. [README.md](README.md)
**What it is**: Original hackathon brief and project overview
**Read this if**: You want to understand the high-level goal
**Reading time**: 5 minutes
**Next**: → [QUICK_START.md](#quick_startmd)

### 2. [QUICK_START.md](QUICK_START.md)
**What it is**: Immediate action items and implementation roadmap
**Read this if**: You're ready to start working NOW
**Reading time**: 10 minutes
**Key sections**:
- What you need to do immediately (data download)
- File organization
- Implementation priority order
- Timeline estimates

**Next**: → [RIGOROUS_METHODOLOGY.md](#rigorous_methodologymd) (Sections 1-2)

---

## 📊 Data Acquisition

### 3. [DATA_DOWNLOAD_SUMMARY.md](data/DATA_DOWNLOAD_SUMMARY.md)
**What it is**: Complete guide to obtaining all required datasets
**Read this if**: Downloads failed, or you need alternative sources
**Reading time**: 15 minutes
**Key sections**:
- Authoritative data sources (DepMap, GDSC)
- Direct download URLs and API endpoints
- File format specifications
- Troubleshooting guide

**Related**: → [data/DATA_SOURCES.md](data/DATA_SOURCES.md) (citations and provenance)

### 4. [scripts/01_download_data.sh](scripts/01_download_data.sh)
**What it is**: Automated download script
**Use this**: To download GDSC data (CCLE requires manual download)
**Run time**: 5-30 minutes depending on connection

**Status Check**:
```bash
bash scripts/01_download_data.sh
# Or check: data_download.log
```

---

## 🔬 Methodology (The Core Documentation)

### 5. [RIGOROUS_METHODOLOGY.md](RIGOROUS_METHODOLOGY.md) ⭐ PRIMARY REFERENCE
**What it is**: Complete methodology from data → validated predictions
**Read this if**: You're implementing the pipeline
**Reading time**: 2 hours (or read sections as needed)

**Table of Contents**:

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
