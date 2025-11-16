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
