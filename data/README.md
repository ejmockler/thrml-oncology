# Data Directory

This directory contains all datasets for the XTR-0 hackathon project.

---

## 📚 Navigation
- **[← Back to Project Root](../DOCUMENTATION_INDEX.md)**
- **[DATA_DOWNLOAD_SUMMARY.md](DATA_DOWNLOAD_SUMMARY.md)** - Complete download guide
- **[DATA_SOURCES.md](DATA_SOURCES.md)** - Citations and provenance

---

## Directory Structure

```
data/
├── README.md                          ← This file
├── DATA_SOURCES.md                    ← Citations, license info
├── DATA_DOWNLOAD_SUMMARY.md           ← Download instructions
│
├── raw/                               ← Raw downloaded data
│   ├── ccle/                          ← CCLE datasets
│   │   ├── CCLE_expression_TPM.csv        ⚠️ NEEDS MANUAL DOWNLOAD
│   │   ├── CCLE_RRBS_methylation.txt      ⚠️ NEEDS MANUAL DOWNLOAD
│   │   └── Model.csv                      ✓ Downloaded (15 KB)
│   │
│   └── gdsc/                          ← GDSC datasets
│       ├── GDSC1_fitted_dose_response.xlsx  ✓ Downloaded (3.3 MB)
│       ├── GDSC2_fitted_dose_response.xlsx  ✓ Downloaded (48 KB)
│       └── Cell_line_RMA_proc_basalExp.txt  ✓ Downloaded (292 MB)
│
└── processed/                         ← Preprocessed outputs
    ├── sensitive_discretized.pkl      (to be created)
    └── resistant_discretized.pkl      (to be created)
```

---

## Download Status

### ✓ Ready to Use (295 MB total) - Automated Downloads Complete

1. **GDSC IC50 Data** (3.3 MB + 48 KB) ✓
   - Source: Sanger Institute Cancer RxGene
   - Files:
     - `GDSC1_fitted_dose_response.xlsx` (3.3 MB)
     - `GDSC2_fitted_dose_response.xlsx` (48 KB)
   - Contains: IC50, AUC, Z-scores for ~400 drugs
   - Status: **Downloaded successfully**

2. **GDSC Expression** (292 MB) ✓
   - Source: GDSC1000 preprocessed
   - File: `Cell_line_RMA_proc_basalExp.txt` (292 MB)
   - Format: RMA-normalized microarray data
   - Purpose: Validation and cross-reference
   - Status: **Downloaded successfully**

3. **CCLE Metadata** (15 KB) ✓
   - Source: DepMap Portal
   - File: `Model.csv` (15 KB)
   - Purpose: ID mapping and filtering
   - Status: **Downloaded successfully**

### ⚠️ Requires Manual Download (2 files needed)

**Why?** DepMap API endpoints require authentication/session management

#### CCLE Expression (~200 MB) - CRITICAL
```bash
# Visit: https://depmap.org/portal/download/
# Search: "OmicsExpressionProteinCodingGenesTPMLogp1"
# Download from: DepMap Public 25Q2 (or latest)
# Save to: data/raw/ccle/CCLE_expression_TPM.csv
```

**Format**: CSV with log2(TPM+1) values
**Dimensions**: ~1,300 cell lines × ~19,000 genes

#### CCLE Methylation (variable size) - CRITICAL
```bash
# Visit: https://depmap.org/portal/download/
# Search: "RRBS" or "CCLE_RRBS_tss1kb"
# Download: Latest RRBS TSS ±1kb file
# Save to: data/raw/ccle/CCLE_RRBS_methylation.txt
```

**Format**: Tab-separated text
**Dimensions**: ~800 cell lines × gene TSS regions
**Values**: β-values (0-1 scale, proportion methylated)

---

## Alternative Sources

### Figshare Archives (Reliable but Older)

If DepMap portal has issues, use these:

**DepMap 23Q4 Release**:
- URL: https://figshare.com/articles/dataset/DepMap_23Q4_Public/24667905
- Contains: CRISPR screens + CCLE genomic characterization
- Note: May not have all omics files separately

**DepMap 23Q2 Release**:
- URL: https://figshare.com/articles/dataset/DepMap_23Q2_Public/22765112
- More complete omics datasets

---

## Verification

After downloading, verify files:

```bash
# Check file sizes
ls -lh data/raw/ccle/
ls -lh data/raw/gdsc/

# Verify expression data
python -c "import pandas as pd; \
  df = pd.read_csv('data/raw/ccle/CCLE_expression_TPM.csv'); \
  print(f'Expression shape: {df.shape}'); \
  print(f'Columns: {list(df.columns[:5])}...')"

# Verify methylation data
python -c "import pandas as pd; \
  df = pd.read_csv('data/raw/ccle/CCLE_RRBS_methylation.txt', sep='\t'); \
  print(f'Methylation shape: {df.shape}')"

# Verify GDSC data
python -c "import pandas as pd; \
  df = pd.read_excel('data/raw/gdsc/GDSC1_fitted_dose_response.xlsx'); \
  print(f'GDSC1 shape: {df.shape}'); \
  print(f'Columns: {list(df.columns)}')"
```

**Expected**:
- Expression: 1000+ rows, 19000+ columns (or transposed)
- Methylation: 800+ rows, variable columns
- GDSC1: 100000+ rows with IC50, AUC, Drug, Cell_Line columns

---

## Data Processing

Once raw data is acquired:

```bash
# Run preprocessing
python core/data_loader.py \
  --ccle-expression data/raw/ccle/CCLE_expression_TPM.csv \
  --ccle-methylation data/raw/ccle/CCLE_RRBS_methylation.txt \
  --gdsc-ic50 data/raw/gdsc/GDSC1_fitted_dose_response.xlsx \
  --output-dir data/processed/

# Output files:
# - data/processed/sensitive_discretized.pkl
# - data/processed/resistant_discretized.pkl
# - data/processed/preprocessing_report.txt
```

See [RIGOROUS_METHODOLOGY.md § 1](../RIGOROUS_METHODOLOGY.md#1-data-preprocessing-pipeline) for details.

---

## Troubleshooting

### "DepMap download returns HTML instead of CSV"
→ API authentication issue
→ Download manually via browser
→ Or use Figshare alternative

### "File is too small (< 1 MB)"
→ Likely an error page
→ Delete and re-download
→ Check file with `head -20 filename`

### "Can't find methylation file on DepMap"
→ Search for "RRBS" or "methylation"
→ Look for "TSS" or "transcription start site"
→ Filter by "CCLE" in file listing

### "Excel file won't open"
→ Install: `pip install openpyxl`
→ Or convert: `libreoffice --convert-to csv file.xlsx`

---

## Citations

When using this data, cite:

**CCLE**:
```
Ghandi, M., Huang, F.W., Jané-Valbuena, J. et al.
Next-generation characterization of the Cancer Cell Line Encyclopedia.
Nature 569, 503–508 (2019).
https://doi.org/10.1038/s41586-019-1186-3
```

**GDSC**:
```
Yang, W., Soares, J., Greninger, P. et al.
Genomics of Drug Sensitivity in Cancer (GDSC):
a resource for therapeutic biomarker discovery in cancer cells.
Nucleic Acids Res. 41, D955-961 (2013).
https://doi.org/10.1093/nar/gks1111
```

Full citations in [DATA_SOURCES.md](DATA_SOURCES.md).

---

## Data Usage Notes

### License
- **CCLE/DepMap**: Academic research use allowed, cite properly
- **GDSC**: Non-commercial research, see website for details
- **This project**: Educational/research hackathon

### Privacy
- All data is de-identified cell lines (not patient data)
- Cell line names are public (ACH-XXXXXX or standard names)

### File Sizes
- Total raw data: ~500 MB (with CCLE files)
- Processed data: ~10-50 MB
- Results: ~1-10 MB

### Storage
- Recommend: External drive or cloud storage for raw data
- Git: Do NOT commit raw data files (too large)
- Add to `.gitignore`: `data/raw/*`, `data/processed/*`

---

## Quick Links

- **Download automation**: [../scripts/01_download_data.sh](../scripts/01_download_data.sh)
- **Preprocessing code**: [../core/data_loader.py](../core/data_loader.py)
- **Full methodology**: [../RIGOROUS_METHODOLOGY.md](../RIGOROUS_METHODOLOGY.md)
- **Troubleshooting**: [DATA_DOWNLOAD_SUMMARY.md](DATA_DOWNLOAD_SUMMARY.md)

---

**Last Updated**: November 16, 2024
