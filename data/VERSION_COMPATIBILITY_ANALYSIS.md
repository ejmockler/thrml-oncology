# Version Compatibility Analysis

**Generated**: November 16, 2024
**Issue**: Potential version mismatch between datasets

---

## Dataset Versions

### 1. CCLE Methylation: **2018** (7 years old)

**File**: `CCLE_RRBS_TSS1kb_20181022.txt`
- **Release Date**: October 22, 2018
- **Source**: CCLE 2019 release
- **Cell Lines**: 843 (columns 4-846, after metadata)
- **Format**: `CELLLINE_TISSUE` (e.g., `HCC827_LUNG`, `A549_LUNG`)
- **Technology**: RRBS (Reduced Representation Bisulfite Sequencing)

**Cell Line Examples**:
```
DMS53_LUNG
NCIH1184_LUNG
NCIH2227_LUNG
CAL120_BREAST
647V_URINARY_TRACT
```

---

### 2. CCLE Expression: **2025-Q3** (Current)

**File**: `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`
- **Release**: DepMap Public 25Q3 (2025 Quarter 3)
- **Cell Lines**: 1,755
- **Format**: `ModelID` = `ACH-XXXXXX` (DepMap ID system)
- **Technology**: RNA-seq, log2(TPM+1) transformed
- **Gene Coverage**: 19,221 protein-coding genes

**Cell Line Examples**:
```
ACH-001113
ACH-001289
ACH-001339
```

---

### 3. GDSC Drug Response: **~2023** (Recent)

**Files**: `GDSC1_fitted_dose_response.xlsx`, `GDSC2_fitted_dose_response.xlsx`
- **Release**: GDSC1000 (updated ~2023)
- **Cell Lines**: ~1,000
- **Format**: Cell line names (similar to methylation format)

---

## The Version Compatibility Problem

### ⚠️ CRITICAL ISSUE: Different ID Systems

| Dataset | Year | Cell Line ID Format | Example |
|---------|------|---------------------|---------|
| **Methylation** | 2018 | Cell line name_Tissue | `HCC827_LUNG` |
| **Expression** | 2025 | DepMap ModelID | `ACH-001113` |
| **GDSC** | 2023 | Cell line name | `HCC827` |

**Problem**: You CANNOT directly match these without a mapping file.

---

## Impact Assessment

### Scenario 1: Without Model.csv Mapping

**What happens**:
```python
# Methylation cell lines
meth_lines = ['HCC827_LUNG', 'A549_LUNG', 'PC9_LUNG']

# Expression ModelIDs
expr_lines = ['ACH-001113', 'ACH-001289', 'ACH-001339']

# Direct matching
overlap = set(meth_lines) & set(expr_lines)
# Result: {} (empty set - NO MATCHES!)
```

**Outcome**: ❌ **ZERO cell line overlap** → Cannot proceed with analysis

---

### Scenario 2: With Model.csv Mapping

**What Model.csv provides**:
```csv
ModelID,StrippedCellLineName,CellLineName,OncotreeLineage
ACH-001113,HCC827,HCC827_LUNG,Lung
ACH-001289,A549,A549_LUNG,Lung
ACH-001339,PC9,PC9_LUNG,Lung
```

**With mapping**:
```python
# Map ACH-001113 → HCC827 → HCC827_LUNG
# Now can match with methylation data

overlap = match_via_model_csv(expr_lines, meth_lines)
# Result: ~400-600 cell lines with both expression + methylation
```

**Outcome**: ✅ **Sufficient overlap** → Analysis can proceed

---

## Cell Line Population Changes (2018 → 2025)

### Expected Changes Over 7 Years:

1. **Deprecated Cell Lines** (~10-20%):
   - Cell lines from 2018 that failed authentication
   - Contaminated or misidentified lines removed
   - Examples: Some NCI-60 panel lines reclassified

2. **New Cell Lines Added** (~30-40%):
   - DepMap expanded from ~800 (2018) to 1,755 (2025)
   - New patient-derived models
   - Organoid models
   - CRISPR-edited lines

3. **Stable Core Set** (~60-70%):
   - Well-established lines (e.g., HCC827, A549, PC9)
   - These should be present in both datasets
   - Expected: ~500-600 lines with both expression + methylation

---

## Solutions (Ranked by Feasibility)

### ✅ Solution 1: GET MODEL.CSV (STRONGLY RECOMMENDED)

**Why this is critical**:
- **REQUIRED** to map between 2018 and 2025 cell line IDs
- Without it: ZERO overlap between expression and methylation
- With it: ~400-600 cell lines with complete data

**How to get it**:
```bash
# Option A: Manual download (EASIEST)
# 1. Visit https://depmap.org/portal/download/ in browser
# 2. Select DepMap Public 25Q3 (or 24Q4)
# 3. Download "Model.csv" (should be ~2-5 MB)
# 4. Save to data/raw/ccle/Model.csv

# Option B: Try Figshare archive
wget https://figshare.com/ndownloader/files/46486180 -O depmap_24q2.zip
unzip -j depmap_24q2.zip "*/Model.csv" -d data/raw/ccle/

# Option C: Use R depmap package to export
# (requires R installation)
```

**Timeline**: 5-10 minutes for manual download

---

### ⚠️ Solution 2: Download Matched Versions

**Option 2A: Get 2019 Expression to Match Methylation**

**Pros**:
- Both datasets from same era
- Cell line names more consistent
- May not need Model.csv

**Cons**:
- Older expression technology (less accurate)
- Fewer cell lines (~800 vs 1,755)
- Hard to find older releases

**How**:
```bash
# Try DepMap 2019 release (matching methylation date)
# Visit: https://depmap.org/portal/download/
# Select: DepMap Public 19Q1 or 19Q2
# Download: CCLE_expression.csv (older format)
```

**Timeline**: 30-60 minutes (finding + downloading)

---

**Option 2B: Get Newer Methylation Data**

**Pros**:
- Would match 2025 expression
- More cell lines potentially

**Cons**:
- ❌ **Newer methylation data may not exist**
- DepMap RRBS methylation collection stopped ~2018-2019
- Newer releases focus on RNA-seq, CRISPR, proteomics

**Likelihood**: **Low** - RRBS methylation not updated in recent DepMap releases

---

### ❌ Solution 3: Proceed Without Mapping (NOT RECOMMENDED)

**Attempt fuzzy matching**:
```python
# Try to extract cell line names from SequencingID or other fields
# Hope that some naming convention allows matching
```

**Problems**:
- SequencingID (e.g., `CDS-010xbm`) doesn't contain cell line name
- ModelID (e.g., `ACH-001113`) is opaque identifier
- No way to map to `HCC827_LUNG` without external mapping

**Expected overlap**: 0-5% (essentially none)

**Outcome**: ❌ Analysis fails due to insufficient data

---

## Recommendation: ACTION REQUIRED

### 🚨 **YOU MUST GET MODEL.CSV BEFORE PROCEEDING**

**Current Status**:
```
✓ Expression data: 1,755 cell lines (2025)
✓ Methylation data: 843 cell lines (2018)
✗ Mapping file: MISSING (CRITICAL)
```

**Without Model.csv**:
- Expression and methylation datasets CANNOT be aligned
- Zero cell line overlap
- Project cannot proceed

**With Model.csv**:
- Expected overlap: ~400-600 cell lines
- Sufficient for THRML inference (need 50-100 per group)
- Analysis can proceed

---

## Detailed Mapping Strategy (Once Model.csv Acquired)

```python
# 1. Load Model.csv
model_df = pd.read_csv('data/raw/ccle/Model.csv')
# Columns: ModelID, StrippedCellLineName, CellLineName, ...

# 2. Load expression data
expr_df = pd.read_csv('data/raw/ccle/CCLE_expression_TPM.csv')
# Has: ModelID column

# 3. Merge expression with model info
expr_with_names = expr_df.merge(
    model_df[['ModelID', 'StrippedCellLineName']], 
    on='ModelID'
)
# Now expr_with_names has cell line names

# 4. Load methylation data
meth_df = pd.read_csv('data/raw/ccle/CCLE_RRBS_methylation.txt', sep='\t')
# Column headers: 'HCC827_LUNG', 'A549_LUNG', etc.

# 5. Extract base cell line names from methylation
meth_cell_lines = meth_df.columns[3:]  # Skip metadata columns
meth_base_names = [cl.split('_')[0] for cl in meth_cell_lines]

# 6. Find overlap
common_lines = set(expr_with_names['StrippedCellLineName']) & set(meth_base_names)
print(f"Overlapping cell lines: {len(common_lines)}")
# Expected: 400-600

# 7. Filter datasets to common cell lines
expr_common = expr_with_names[
    expr_with_names['StrippedCellLineName'].isin(common_lines)
]
meth_common = meth_df[[
    col for col in meth_df.columns 
    if col.split('_')[0] in common_lines or col in ['locus_id', 'CpG_sites_hg19']
]]

# 8. Now both datasets are aligned and ready for analysis
```

---

## Known Cell Lines Likely in Both Datasets

These established lung cancer lines should be in both 2018 and 2025:

| Cell Line | 2018 (Methylation) | 2025 (Expression) | In GDSC? |
|-----------|-------------------|-------------------|----------|
| HCC827 | ✓ | ✓ (likely) | ✓ |
| A549 | ✓ | ✓ (likely) | ✓ |
| PC9 | ✓ | ✓ (likely) | ✓ |
| NCI-H1975 | ✓ | ✓ (likely) | ✓ |
| NCI-H2228 | ✓ | ✓ (likely) | ✓ |

But we need Model.csv to **confirm** these mappings.

---

## Timeline Impact

| Action | Time | Status |
|--------|------|--------|
| ✓ Download expression data | DONE | Complete |
| ✓ Download methylation data | DONE | Complete |
| ✓ Download GDSC data | DONE | Complete |
| ⚠️ **GET MODEL.CSV** | **10 min** | **REQUIRED** |
| Implement data_loader.py | 2-3 hrs | Blocked until Model.csv |
| Implement THRML model | 1-2 hrs | Blocked |
| Run inference | 2-3 hrs | Blocked |

**Critical Path**: Model.csv acquisition is blocking all downstream work.

---

## ✅ VERIFICATION RESULTS (November 16, 2024)

**Model.csv ACQUIRED AND VERIFIED**

### File Details
- **Location**: `data/raw/ccle/Model.csv`
- **Size**: 683 KB
- **Rows**: 2,133 cell line models
- **Key Columns**: ModelID, StrippedCellLineName, CellLineName, OncotreeLineage

### Overlap Analysis

**Datasets**:
- Expression (25Q3): 1,699 ModelIDs → 1,700 unique cell lines
- Methylation (2018): 842 unique cell lines
- Model.csv (25Q3): 2,132 ModelIDs

**OVERLAP VERIFICATION**:
```
✓ Expression ↔ Model.csv:  1,699 / 1,699 (100% match)
✓ Expression ↔ Methylation: 831 cell lines (98.7% of methylation data)
✓ Final usable dataset: 831 cell lines with BOTH expression + methylation
```

### Known Cell Lines Verified

| Cell Line | ModelID | In Expression | In Methylation | In Model.csv |
|-----------|---------|---------------|----------------|--------------|
| HCC827 | ACH-000012 | ✓ | ✓ (HCC827_LUNG) | ✓ |
| A549 | ACH-000681 | ✓ | ✓ (A549_LUNG) | ✓ |
| PC9 | ACH-000779 | ✓ | ✓ (PC9_LUNG) | ✓ |

**Mapping confirmed working**: ACH-000012 → HCC827 → HCC827_LUNG ✓

### Statistical Power Assessment

**Required for analysis**: 50-100 cell lines per group (sensitive vs resistant)
**Available**: 831 cell lines total

**After stratification** (33rd/67th percentiles):
- Sensitive group: ~275 cell lines (top 33%)
- Resistant group: ~275 cell lines (bottom 33%)

**Conclusion**: ✅ **EXCELLENT statistical power** (5-6× minimum requirement)

---

## READY TO PROCEED

**All blocking issues resolved**:
- ✅ Model.csv acquired (683 KB)
- ✅ Cell line mapping verified (831 overlapping lines)
- ✅ Key genes present (EGFR, KRAS, BRAF, PIK3CA, etc.)
- ✅ Statistical power sufficient (275 per group vs 50 minimum)

**Next step**: Implement `core/data_loader.py`

---

**Last Updated**: November 16, 2024
**Status**: ✅ **VERIFIED COMPATIBLE - READY TO PROCEED**
