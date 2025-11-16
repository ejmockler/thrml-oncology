# Data Download Summary for XTR-0 Hackathon

**Date**: November 16, 2024
**Project**: Thermodynamic Causal Inference for Drug Response Prediction

## Downloaded Data Status

### ✓ GDSC (Genomics of Drug Sensitivity in Cancer)

#### 1. Drug Response Data (IC50)
- **File**: `data/raw/gdsc/GDSC_drug_data_fitted.xlsx`
- **Size**: 3.3 MB
- **Source**: Sanger Institute Cancer RxGene
- **URL**: https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Data/suppData/TableS4A.xlsx
- **Description**: Fitted dose-response curves with IC50, AUC, and Z-scores
- **Contains**: ~400 drugs × ~1,000 cell lines

#### 2. Expression Data
- **File**: `data/raw/gdsc/Cell_line_RMA_proc_basalExp.txt`
- **Size**: 292 MB
- **Source**: GDSC1000 preprocessed data
- **Format**: RMA-normalized microarray expression
- **Purpose**: Validation and cross-reference with CCLE

### ⚠️ CCLE (Cancer Cell Line Encyclopedia) - NEEDS COMPLETION

#### Currently Downloaded:
- **File**: `data/raw/ccle/CCLE_expression.csv` (15 KB - INCOMPLETE!)
  - This appears to be an error page, not actual data

#### Still Needed:
1. **CCLE Expression Data** (high priority)
   - Target file: `OmicsExpressionProteinCodingGenesTPMLogp1.csv`
   - Expected size: ~100-250 MB
   - Format: CSV with log2(TPM+1) values
   - Genes: ~19,000 protein-coding genes
   - Cell lines: ~1,300 cancer cell lines

2. **CCLE Methylation Data** (high priority)
   - Target file: `CCLE_RRBS_tss1kb_*.txt`
   - Method: RRBS (Reduced Representation Bisulfite Sequencing)
   - Regions: TSS ±1kb
   - Cell lines: ~800+ cell lines

3. **CCLE Metadata** (medium priority)
   - Target file: `Model.csv`
   - Purpose: Map cell line identifiers and annotations

## Authoritative Data Sources

### DepMap Portal (Broad Institute)
- **Main Portal**: https://depmap.org/portal/download/
- **Latest Release**: DepMap Public 25Q2 (most recent as of Nov 2024)
- **API Base**: `https://depmap.org/portal/download/api/download/external`

**Key Files Available**:

1. **Expression** (RNA-seq TPM):
```bash
https://depmap.org/portal/download/api/download/external?file_name=ccle%2Fdepmap-public-cell-line-expression-25q2-c23f.18%2FOmicsExpressionProteinCodingGenesTPMLogp1.csv&bucket=depmap-external-downloads
```

2. **Methylation** (RRBS):
```bash
https://depmap.org/portal/download/api/download/external?file_name=ccle%2Fdepmap-rrbs-v3-1-7ea8.6%2FCCLE_RRBS_tss1kb_20181022.txt&bucket=depmap-external-downloads
```

3. **Metadata**:
```bash
https://depmap.org/portal/download/api/download/external?file_name=ccle%2Fdepmap-public-cell-line-metadata-e68d.37%2FModel.csv&bucket=depmap-external-downloads
```

### Alternative: Figshare Archives
- **DepMap 23Q4**: https://figshare.com/articles/dataset/DepMap_23Q4_Public/24667905
  - Note: 23Q4 may not contain all omics datasets
- **DepMap 23Q2**: https://figshare.com/articles/dataset/DepMap_23Q2_Public/22765112
- **DepMap 21Q4**: https://figshare.com/articles/dataset/DepMap_21Q4_Public/16924132

### GDSC Portal (Sanger Institute)
- **Main Portal**: https://www.cancerrxgene.org/
- **Downloads**: https://www.cancerrxgene.org/downloads

**Files Used**:
- IC50 Data: `TableS4A.xlsx` (GDSC1)
- IC50 Data: `TableS4E.xlsx` (GDSC2)
- Expression: `Cell_line_RMA_proc_basalExp.txt.zip`

## Next Steps

### Immediate Actions Required:

1. **Download CCLE Expression Data**:
   ```bash
   curl -L "https://depmap.org/portal/download/api/download/external?file_name=ccle%2Fdepmap-public-cell-line-expression-25q2-c23f.18%2FOmicsExpressionProteinCodingGenesTPMLogp1.csv&bucket=depmap-external-downloads" \
     -o data/raw/ccle/CCLE_expression_TPM.csv
   ```

2. **Download CCLE Methylation Data**:
   ```bash
   curl -L "https://depmap.org/portal/download/api/download/external?file_name=ccle%2Fdepmap-rrbs-v3-1-7ea8.6%2FCCLE_RRBS_tss1kb_20181022.txt&bucket=depmap-external-downloads" \
     -o data/raw/ccle/CCLE_RRBS_methylation.txt
   ```

3. **Download CCLE Metadata**:
   ```bash
   curl -L "https://depmap.org/portal/download/api/download/external?file_name=ccle%2Fdepmap-public-cell-line-metadata-e68d.37%2FModel.csv&bucket=depmap-external-downloads" \
     -o data/raw/ccle/Model.csv
   ```

### Or Run the Complete Script:

```bash
bash scripts/01_download_data.sh
```

This script will:
- Download all required CCLE and GDSC data
- Verify file integrity
- Extract compressed files
- Create documentation

## Data Requirements for Hackathon

### Minimum Viable Demo:
- ✓ GDSC IC50 data (for validation)
- ⚠️ CCLE Expression data (CRITICAL - for causal inference)
- ⚠️ CCLE Methylation data (CRITICAL - for causal anchoring)
- ○ CCLE Metadata (helpful for filtering)

### For Full Pipeline:
- All of the above
- Focus on EGFR pathway genes (~10-20 genes)
- Erlotinib-resistant vs sensitive cell lines
- Discretize continuous values to {low, med, high}

## Data Specifications

### CCLE Expression (Expected)
- **Format**: CSV
- **Dimensions**: ~1,300 cell lines × ~19,000 genes
- **Values**: log2(TPM+1) - already normalized and logged
- **Header**: Gene symbols in columns, cell lines in rows (or transposed)

### CCLE Methylation (Expected)
- **Format**: Tab-separated text file
- **Dimensions**: ~800 cell lines × gene promoter regions
- **Values**: Beta values (0.0 - 1.0, proportion methylated)
- **Regions**: TSS ±1kb (transcription start sites)

### GDSC IC50 (Downloaded)
- **Format**: Excel (.xlsx)
- **Columns**: Cell line, Drug, IC50, AUC, RMSE, Z-score
- **Drugs**: Includes erlotinib and alternatives
- **Purpose**: Ground truth for validation

## Citations

When using this data, cite:

**DepMap/CCLE**:
- Ghandi, M., Huang, F.W., Jané-Valbuena, J. et al. Next-generation characterization of the Cancer Cell Line Encyclopedia. Nature 569, 503–508 (2019).

**GDSC**:
- Yang, W., Soares, J., Greninger, P. et al. Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells. Nucleic Acids Res. 41, D955-961 (2013).

**RRBS/Methylation**:
- Barretina, J., Caponigro, G., Stransky, N. et al. The Cancer Cell Line Encyclopedia enables predictive modelling of anticancer drug sensitivity. Nature 483, 603–607 (2012).

## Troubleshooting

### If Downloads Fail:

1. **Check network connectivity**: DepMap/GDSC servers may be slow
2. **Verify URLs**: API endpoints may change between releases
3. **Try Figshare**: Often more stable for large files
4. **Use wget instead of curl**: `wget -O <output> <url>`
5. **Download in browser**: Visit URLs directly and save manually

### If Files Are Corrupt:

1. **Check file size**: Compare with expected sizes above
2. **Verify format**: Use `head -20 <file>` to inspect
3. **Re-download**: Delete and try again
4. **Check MD5/checksums**: If provided by source

## Contact and Support

- **DepMap Help**: depmap@broadinstitute.org
- **DepMap Forum**: https://forum.depmap.org/
- **GDSC Contact**: Through website contact form
- **This Project**: See README.md for hackathon details
