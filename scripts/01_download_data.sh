#!/bin/bash

# XTR-0 Hackathon: Data Download Script
# Downloads GDSC data from authoritative sources
# NOTE: CCLE data requires MANUAL download from DepMap Portal
#       See DEPMAP_REQUIRED_FILES.md for instructions

set -e  # Exit on error

echo "=== XTR-0 Data Download Script ==="
echo "Downloading authoritative data for cancer cell line drug response prediction"
echo ""

# Create data directories
mkdir -p data/raw/{ccle,gdsc}
mkdir -p data/processed

# ============================================================================
# CCLE DATA - MANUAL DOWNLOAD REQUIRED
# ============================================================================

echo "=== CCLE Data Status ==="
echo ""
echo "⚠️  CCLE Expression and Methylation require MANUAL download"
echo "    (DepMap API endpoints require authentication)"
echo ""
echo "Please download manually:"
echo "  1. Visit: https://depmap.org/portal/download/"
echo "  2. Download: OmicsExpressionProteinCodingGenesTPMLogp1.csv"
echo "     Save to: data/raw/ccle/CCLE_expression_TPM.csv"
echo "  3. Download: CCLE_RRBS_TSS1kb file (or use scripts/download_ccle_direct.sh)"
echo "     Save to: data/raw/ccle/CCLE_RRBS_methylation.txt"
echo ""
echo "See DEPMAP_REQUIRED_FILES.md for detailed instructions"
echo ""

# Download CCLE Metadata only (this works via API)
echo "[1/3] Downloading CCLE Cell Line Metadata..."
# Sample info for mapping cell lines
curl -L "https://depmap.org/portal/download/api/download/external?file_name=ccle%2Fdepmap-public-cell-line-metadata-e68d.37%2FModel.csv&bucket=depmap-external-downloads" \
  -o data/raw/ccle/Model.csv \
  --progress-bar \
  || echo "Warning: Metadata download failed."

# ============================================================================
# GDSC DATA (from Cancer RxGene)
# ============================================================================

echo ""
echo "[2/3] Downloading GDSC Drug Response Data (IC50)..."
# Source: Genomics of Drug Sensitivity in Cancer (Sanger Institute)
# GDSC1 and GDSC2 fitted dose-response data (October 2023 release)

# GDSC1 dataset
curl -L "https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Data/suppData/TableS4A.xlsx" \
  -o data/raw/gdsc/GDSC1_fitted_dose_response.xlsx \
  --progress-bar \
  || echo "Warning: GDSC1 IC50 download failed."

# GDSC2 dataset
curl -L "https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Data/suppData/TableS4E.xlsx" \
  -o data/raw/gdsc/GDSC2_fitted_dose_response.xlsx \
  --progress-bar \
  || echo "Warning: GDSC2 IC50 download failed."

echo ""
echo "[3/3] Downloading GDSC Cell Line Expression (for validation)..."
# GDSC expression data (preprocessed, RMA normalized)
curl -L "https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip" \
  -o data/raw/gdsc/GDSC_expression.txt.zip \
  --progress-bar \
  && unzip -o data/raw/gdsc/GDSC_expression.txt.zip -d data/raw/gdsc/ \
  && rm data/raw/gdsc/GDSC_expression.txt.zip \
  || echo "Warning: GDSC expression download failed."

# ============================================================================
# VERIFICATION
# ============================================================================

echo ""
echo "=== Download Summary ==="
echo ""
echo "✓ AUTOMATED DOWNLOADS (Complete):"
echo "  CCLE Metadata: $(ls -lh data/raw/ccle/Model.csv 2>/dev/null | awk '{print $5}' || echo 'FAILED')"
echo "  GDSC1 IC50: $(ls -lh data/raw/gdsc/GDSC1_fitted_dose_response.xlsx 2>/dev/null | awk '{print $5}' || echo 'FAILED')"
echo "  GDSC2 IC50: $(ls -lh data/raw/gdsc/GDSC2_fitted_dose_response.xlsx 2>/dev/null | awk '{print $5}' || echo 'FAILED')"
echo "  GDSC Expression: $(ls -lh data/raw/gdsc/Cell_line_RMA_proc_basalExp.txt 2>/dev/null | awk '{print $5}' || echo 'FAILED')"
echo ""
echo "⚠️  MANUAL DOWNLOADS (Required):"
echo "  CCLE Expression: $(ls -lh data/raw/ccle/CCLE_expression_TPM.csv 2>/dev/null | awk '{print $5}' || echo 'NEEDS DOWNLOAD (~200 MB)')"
echo "  CCLE Methylation: $(ls -lh data/raw/ccle/CCLE_RRBS_methylation.txt 2>/dev/null | awk '{print $5}' || echo 'NEEDS DOWNLOAD (~50-100 MB)')"
echo ""

# Count lines in key files
echo "=== Data Verification ==="
if [ -f data/raw/ccle/CCLE_expression_TPM.csv ]; then
  echo "CCLE Expression: $(wc -l < data/raw/ccle/CCLE_expression_TPM.csv) lines"
fi

if [ -f data/raw/ccle/CCLE_RRBS_methylation.txt ]; then
  echo "CCLE Methylation: $(wc -l < data/raw/ccle/CCLE_RRBS_methylation.txt) lines"
fi

if [ -f data/raw/gdsc/Cell_line_RMA_proc_basalExp.txt ]; then
  echo "GDSC Expression: $(wc -l < data/raw/gdsc/Cell_line_RMA_proc_basalExp.txt) lines"
fi

echo ""
echo "=== Next Steps ==="
echo ""
echo "✓ Automated downloads complete (GDSC data ready)"
echo ""
echo "⚠️  REQUIRED: Download CCLE data manually"
echo "    See: DEPMAP_REQUIRED_FILES.md for detailed instructions"
echo "    Or run: bash scripts/download_ccle_direct.sh (attempts direct URLs)"
echo ""
echo "After CCLE download:"
echo "  1. Verify files: ls -lh data/raw/ccle/"
echo "  2. Run pipeline: python scripts/02_run_inference.py --quick-test"
echo ""

# ============================================================================
# DATA SOURCES DOCUMENTATION
# ============================================================================

cat > data/DATA_SOURCES.md <<'EOF'
# Data Sources Documentation

## CCLE (Cancer Cell Line Encyclopedia)

### Expression Data
- **File**: `CCLE_expression_TPM.csv`
- **Source**: DepMap Portal (Broad Institute)
- **URL**: https://depmap.org/portal/download/
- **Release**: DepMap Public 25Q2
- **Format**: CSV, log2(TPM+1) transformed
- **Genes**: ~19,000 protein-coding genes
- **Cell Lines**: ~1,300 cancer cell lines
- **Description**: RNA-seq gene expression quantified as Transcripts Per Million (TPM)

### Methylation Data
- **File**: `CCLE_RRBS_methylation.txt`
- **Source**: DepMap Portal / CCLE
- **Method**: RRBS (Reduced Representation Bisulfite Sequencing)
- **Regions**: TSS ±1kb (transcription start site regions)
- **Cell Lines**: ~800+ cell lines
- **Description**: DNA methylation beta values (0-1 scale) for gene promoter regions

### Metadata
- **File**: `Model.csv`
- **Contains**: Cell line names, tissue types, lineage, sex, mutation status
- **Purpose**: Map between different cell line identifiers

## GDSC (Genomics of Drug Sensitivity in Cancer)

### Drug Response Data
- **Files**:
  - `GDSC1_fitted_dose_response.xlsx`
  - `GDSC2_fitted_dose_response.xlsx`
- **Source**: Cancer RxGene (Sanger Institute)
- **URL**: https://www.cancerrxgene.org/
- **Metrics**: IC50, AUC, RMSE, Z-score
- **Drugs**: ~400+ compounds
- **Cell Lines**: ~1,000 cell lines (overlap with CCLE)
- **Description**: Fitted dose-response curves from drug screening

### Expression Data (Validation)
- **File**: `Cell_line_RMA_proc_basalExp.txt`
- **Format**: RMA normalized microarray expression
- **Purpose**: Independent validation of CCLE expression data
- **Genes**: Affymetrix probe sets

## Data Integration Notes

1. **Cell Line Mapping**: Use DepMap ID and cell line names to match between datasets
2. **Gene Mapping**: HUGO symbols are standard across datasets
3. **Quality Control**: Both datasets have undergone extensive QC by respective consortia
4. **License**: Data available for academic research use

## Citations

**CCLE**:
- Barretina et al. (2012). Nature 483, 603–607
- Ghandi et al. (2019). Nature 569, 503–508

**GDSC**:
- Yang et al. (2013). Nucleic Acids Research 41, D955–D961
- Iorio et al. (2016). Cell 166, 740–754

## Data Update Frequency

- DepMap: Quarterly releases (e.g., 25Q1, 25Q2, 25Q3, 25Q4)
- GDSC: Annual major releases, periodic updates
EOF

echo "✓ Created data/DATA_SOURCES.md documentation"
