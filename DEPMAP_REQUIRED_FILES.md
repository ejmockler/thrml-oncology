# DepMap Required Files - Exact Specifications

**Last Updated**: November 16, 2024

---

## ⚠️ MANUAL DOWNLOAD REQUIRED

**Automated download failed**: DepMap API endpoints require authentication/session management.

**Status**: The automated download script (`scripts/01_download_data.sh`) successfully downloaded:
- ✓ GDSC drug response data (3.3 MB + 48 KB)
- ✓ GDSC expression data (292 MB)
- ✓ CCLE metadata (15 KB)

**But CCLE expression and methylation must be downloaded manually** (see instructions below).

---

## Critical Files Needed from DepMap

We need **exactly 2 files** from the DepMap Portal to complete the dataset:

---

## 1. CCLE Expression Data (RNA-seq TPM) ⚠️ CRITICAL

### What We Need
**Gene expression levels** for cancer cell lines using RNA-seq quantification.

### Exact File Specifications

**File Name Options** (use latest available):
- `OmicsExpressionProteinCodingGenesTPMLogp1.csv`
- `CCLE_expression.csv` (older naming)
- `OmicsExpressionProteinCodingGenesTPM.csv`

**Release**: DepMap Public **25Q2** (or latest: 25Q3, 25Q4)

**Format**: CSV
**Size**: ~100-250 MB (uncompressed)
**Dimensions**: Approximately 1,300 cell lines × 19,000 genes

**Data Type**:
- log₂(TPM + 1) transformed values
- TPM = Transcripts Per Million (RNA-seq quantification)
- Already normalized and log-transformed by DepMap

**Content Structure**:
```
Expected columns: Gene symbols (EGFR, KRAS, etc.)
Expected rows: Cell line IDs (ACH-000001, etc.) or DepMap IDs
Values: Floating point, typically range 0-15
```

### How to Download

**Method 1: DepMap Portal (Recommended)**
```
1. Visit: https://depmap.org/portal/download/
2. Select release: "DepMap Public 25Q2" (or latest)
3. Find section: "Expression" or "Omics"
4. Look for: File name containing "Expression" and "TPM" or "Logp1"
5. Download: Click download button
6. Save as: data/raw/ccle/CCLE_expression_TPM.csv
```

**Method 2: Direct API URL** (may require authentication)
```bash
# Template URL (update release version)
https://depmap.org/portal/download/api/download/external?\
file_name=ccle%2Fdepmap-public-cell-line-expression-25q2-c23f.18%2FOmicsExpressionProteinCodingGenesTPMLogp1.csv&\
bucket=depmap-external-downloads

# Save to:
curl -L "<URL>" -o data/raw/ccle/CCLE_expression_TPM.csv
```

**Method 3: Figshare (Older but Reliable)**
```
DepMap 23Q4: https://figshare.com/articles/dataset/DepMap_23Q4_Public/24667905
Look for: CCLE_expression.csv or similar in file listing
```

### Why We Need This
- **Primary data**: Expression levels for 15-20 EGFR pathway genes
- **Cell line stratification**: Identify which genes are expressed in each cell line
- **Discretization input**: Convert continuous expression → {low, med, high}
- **Network inference**: E_g ↔ E_h interactions

### Usage in Pipeline
```python
# In core/data_loader.py
expression_df = pd.read_csv('data/raw/ccle/CCLE_expression_TPM.csv', index_col=0)

# Extract EGFR pathway genes
target_genes = ['EGFR', 'KRAS', 'BRAF', 'PIK3CA', ...]
expression_subset = expression_df[target_genes]

# Align with cell lines that have drug response data
common_lines = set(expression_df.index) & set(gdsc_cell_lines)
expression_final = expression_subset.loc[common_lines]

# Discretize (tertile binning)
expression_discrete = discretize_tertiles(expression_final)
```

---

## 2. CCLE Methylation Data (RRBS) ⚠️ CRITICAL

### What We Need
**DNA methylation levels** at gene promoter regions (TSS ±1kb).

### Exact File Specifications

**File Name** (use this specific file):
- `CCLE_RRBS_TSS1kb_20181022.txt.gz` ⭐ **RECOMMENDED**
- Alternative: `CCLE_RRBS_TSS1kb_20181022.txt` (uncompressed)

**NOT recommended** (less suitable for our analysis):
- ❌ `CCLE_RRBS_tss_CpG_clusters_20181022.txt.gz` (105 MB, requires aggregation)
- ❌ `CCLE_RRBS_cgi_CpG_clusters_20181119.txt.gz` (156 MB, CpG islands not promoters)
- ❌ `CCLE_RRBS_enh_CpG_clusters_20181119.txt.gz` (2.4 MB, enhancers not promoters)

**Why TSS ±1kb?** Strongest causal signal for M → E relationship (see methodology docs)

**Release**: Methylation (RRBS) release (separate from quarterly releases)

**Format**: Tab-separated text (.txt)
**Size**: Variable (may be compressed)
**Dimensions**: ~800-900 cell lines × gene TSS regions

**Data Type**:
- β-values (beta values)
- Range: 0.0 to 1.0
- Interpretation: Proportion of CpG sites methylated
- 0.0 = unmethylated, 1.0 = fully methylated

**Content Structure**:
```
Expected columns: Gene_TSS_region identifiers (e.g., EGFR_chr7_55019032_55020032)
Expected rows: Cell line names or IDs
Values: Floating point, range 0.0-1.0
```

### How to Download

**Method 1: DepMap Portal**
```
1. Visit: https://depmap.org/portal/download/
2. Select: "Methylation (RRBS)" from release dropdown
3. Look for: File with "RRBS" and "TSS" or "tss1kb" in name
4. Download
5. Save as: data/raw/ccle/CCLE_RRBS_methylation.txt
```

**Method 2: Direct API URL**
```bash
# Known URL (may need to update version)
https://depmap.org/portal/download/api/download/external?\
file_name=ccle%2Fdepmap-rrbs-v3-1-7ea8.6%2FCCLE_RRBS_tss1kb_20181022.txt&\
bucket=depmap-external-downloads

# Save to:
curl -L "<URL>" -o data/raw/ccle/CCLE_RRBS_methylation.txt
```

**Method 3: Legacy CCLE Site**
```
Visit: https://sites.broadinstitute.org/ccle/datasets
Section: "DNA Methylation (RRBS)"
Download the TSS ±1kb summarized file
```

### Why We Need This
- **Causal anchor**: Methylation → Expression (M causes E, not vice versa)
- **Discretization**: Convert β-values → {low, med, high}
- **Model input**: M_g → E_g factors in energy function
- **Breaking symmetry**: Distinguishes causal direction in inference

### Usage in Pipeline
```python
# In core/data_loader.py
methylation_df = pd.read_csv('data/raw/ccle/CCLE_RRBS_methylation.txt',
                              sep='\t', index_col=0)

# Map gene names to TSS columns
# (TSS columns may be: EGFR_chr7_... format)
for gene in target_genes:
    tss_cols = [c for c in methylation_df.columns if gene in c and 'TSS' in c]
    if tss_cols:
        # Average multiple CpG sites
        methylation_data[gene] = methylation_df[tss_cols].mean(axis=1)

# Align with expression data
methylation_aligned = methylation_data.loc[common_lines]

# Discretize (tertile or 0.3/0.7 thresholds)
methylation_discrete = discretize_tertiles(methylation_aligned)
```

---

## 3. Cell Line Metadata (Model.csv) ✓ ALREADY DOWNLOADED

**File**: `Model.csv` (15 KB)
**Status**: Already downloaded by scripts/01_download_data.sh
**Location**: `data/raw/ccle/Model.csv`

**Purpose**: Maps cell line identifiers and provides tissue annotations
**No action needed** - this file is ready to use.

---

## Summary: What to Download NOW

### Required Actions

1. **Go to DepMap Portal**:
   ```
   https://depmap.org/portal/download/
   ```

2. **Download File #1: Expression**
   - Search for: "Expression" + "TPM" or "Logp1"
   - Or filter by: DepMap Public 25Q2
   - File type: CSV
   - Expected size: 100-250 MB
   - Save as: `data/raw/ccle/CCLE_expression_TPM.csv`

3. **Download File #2: Methylation**
   - Search for: "RRBS" or "methylation"
   - Or select: "Methylation (RRBS)" release
   - Look for: "TSS" or "tss1kb" in filename
   - File type: TXT (tab-separated)
   - Save as: `data/raw/ccle/CCLE_RRBS_methylation.txt`

---

## Verification After Download

Run these commands to verify files are correct:

```bash
# Check file sizes
ls -lh data/raw/ccle/CCLE_expression_TPM.csv
ls -lh data/raw/ccle/CCLE_RRBS_methylation.txt

# Should be:
# Expression: 100-250 MB
# Methylation: Variable (50-200 MB)
```

```python
# Verify expression file
import pandas as pd

expr = pd.read_csv('data/raw/ccle/CCLE_expression_TPM.csv', index_col=0)
print(f"Expression shape: {expr.shape}")
print(f"Genes present: {expr.shape[1]}")
print(f"Cell lines: {expr.shape[0]}")
print(f"Has EGFR: {'EGFR' in expr.columns or 'EGFR' in expr.index}")

# Expected output:
# Shape: ~(1300, 19000) or ~(19000, 1300) if transposed
# Has EGFR: True
```

```python
# Verify methylation file
meth = pd.read_csv('data/raw/ccle/CCLE_RRBS_methylation.txt',
                   sep='\t', index_col=0)
print(f"Methylation shape: {meth.shape}")
print(f"Value range: {meth.min().min():.2f} to {meth.max().max():.2f}")
print(f"Has EGFR TSS: {any('EGFR' in c for c in meth.columns)}")

# Expected output:
# Shape: ~(800, variable)
# Value range: 0.00 to 1.00 (beta values)
# Has EGFR TSS: True
```

---

## Troubleshooting

### "I can't find the files on DepMap"

**Solution 1**: Use search box
```
Type: "OmicsExpression" or "TPM"
Type: "RRBS" or "methylation"
```

**Solution 2**: Filter by release
```
Click: "DepMap Public 25Q2" dropdown
Select: Latest quarterly release
Browse: Files should be listed
```

**Solution 3**: Use Figshare
```
Visit: https://figshare.com/articles/dataset/DepMap_23Q4_Public/24667905
Download: Files directly from browser
Note: May be older release (23Q4 vs 25Q2)
```

### "Download fails or returns HTML"

**Issue**: API authentication required

**Solution**: Download via browser
```
1. Visit DepMap portal in browser
2. Click download button (don't use curl)
3. Save file manually
4. Move to data/raw/ccle/
```

### "File format looks wrong"

**Check**:
```bash
# First 20 lines
head -20 data/raw/ccle/CCLE_expression_TPM.csv

# Should see:
# - Header row with gene names OR cell line IDs
# - Data rows with numeric values
# - CSV format (comma-separated)

# If you see HTML tags, it's an error page
```

**Fix**: Delete and re-download

### "Do I need to decompress files?"

**Expression**: Usually downloaded as .csv (already uncompressed)

**Methylation**: May be .txt or .txt.gz
```bash
# If compressed (.gz)
gunzip data/raw/ccle/CCLE_RRBS_methylation.txt.gz
```

---

## Alternative: Use Older Releases

If latest releases unavailable, these older versions work:

### DepMap 23Q4 (Figshare)
```
URL: https://figshare.com/articles/dataset/DepMap_23Q4_Public/24667905
Files: Browse file listing for:
  - CCLE_expression.csv
  - CCLE mutations (may be in place of methylation)
```

### DepMap 21Q4 (Figshare)
```
URL: https://figshare.com/articles/dataset/DepMap_21Q4_Public/16924132
Note: Older data but complete omics sets
```

### Legacy CCLE Site
```
URL: https://sites.broadinstitute.org/ccle/datasets
Sections:
  - RNAseq Gene Expression
  - DNA Methylation (RRBS)
Note: May be older versions
```

---

## Why These Specific Files?

### Expression (log2 TPM+1)
- **Already normalized**: No need for complex preprocessing
- **Log-transformed**: Reduces dynamic range for discretization
- **Standard format**: TPM is industry standard for RNA-seq

### Methylation (RRBS TSS ±1kb)
- **Promoter regions**: TSS ±1kb captures regulatory methylation
- **Gene-level**: Summarized per gene (not individual CpGs)
- **β-values**: Direct interpretation as proportion methylated

### Not needed
- **Mutations**: We're using expression + methylation only
- **Copy number**: Not required for this analysis
- **Proteomics**: RNA-seq sufficient for gene expression
- **CRISPR screens**: Not used in this pipeline

---

## Data Flow Reminder

```
DepMap Downloads
├─► CCLE_expression_TPM.csv (log2 TPM+1)
│   └─► Extract 15 EGFR genes × 1300 cell lines
│       └─► Align with GDSC drug response (50-100 lines per group)
│           └─► Discretize → {0, 1, 2} = {low, med, high}
│               └─► Input to THRML CategoricalNode (E_g)
│
└─► CCLE_RRBS_methylation.txt (β-values)
    └─► Extract TSS regions for same 15 genes
        └─► Align with same cell lines
            └─► Discretize → {0, 1, 2} = {low, med, high}
                └─► Input to THRML CategoricalNode (M_g)

THRML Model:
M_g → E_g (methylation influences expression)
E_g ↔ E_h (gene interactions)

Inference:
Compare networks: sensitive vs resistant
Identify bypass edges
Predict drugs
Validate with GDSC IC50
```

---

## Quick Checklist

- [ ] Visit https://depmap.org/portal/download/
- [ ] Download: OmicsExpressionProteinCodingGenesTPMLogp1.csv (or similar)
- [ ] Save as: `data/raw/ccle/CCLE_expression_TPM.csv`
- [ ] Download: CCLE_RRBS_tss1kb file (or similar)
- [ ] Save as: `data/raw/ccle/CCLE_RRBS_methylation.txt`
- [ ] Verify file sizes (100+ MB each)
- [ ] Run verification Python scripts above
- [ ] Confirm files have gene names (EGFR, KRAS, etc.)

**Once complete**: Proceed to `core/data_loader.py` implementation

---

**Last Updated**: November 16, 2024
**For More Info**: See [DATA_DOWNLOAD_SUMMARY.md](data/DATA_DOWNLOAD_SUMMARY.md)
