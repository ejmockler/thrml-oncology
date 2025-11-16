#!/bin/bash
# Direct CCLE Data Download with Exact URLs
# Run this if the main download script fails

set -e

echo "=== Direct CCLE Data Download ==="
echo ""

# Create directories
mkdir -p data/raw/ccle

# ============================================================================
# CCLE Expression (25Q2 or latest)
# ============================================================================

echo "[1/2] Downloading CCLE Expression (RNA-seq TPM log2+1)..."
echo "This may take 5-10 minutes (100-250 MB file)"

# Try 25Q2 first
curl -L "https://depmap.org/portal/download/all/?release=DepMap+Public+25Q2&file=OmicsExpressionProteinCodingGenesTPMLogp1.csv" \
  -o data/raw/ccle/CCLE_expression_TPM.csv \
  --progress-bar \
  --max-time 600 \
  || echo "Warning: 25Q2 download failed, trying 24Q4..."

# If 25Q2 failed, try 24Q4
if [ ! -s data/raw/ccle/CCLE_expression_TPM.csv ] || [ $(wc -c < data/raw/ccle/CCLE_expression_TPM.csv) -lt 1000000 ]; then
  echo "Trying 24Q4 release..."
  curl -L "https://depmap.org/portal/download/all/?release=DepMap+Public+24Q4&file=OmicsExpressionProteinCodingGenesTPMLogp1.csv" \
    -o data/raw/ccle/CCLE_expression_TPM.csv \
    --progress-bar \
    --max-time 600 \
    || echo "Warning: 24Q4 also failed, trying 24Q2..."
fi

# If still failed, try 24Q2
if [ ! -s data/raw/ccle/CCLE_expression_TPM.csv ] || [ $(wc -c < data/raw/ccle/CCLE_expression_TPM.csv) -lt 1000000 ]; then
  echo "Trying 24Q2 release..."
  curl -L "https://depmap.org/portal/download/all/?release=DepMap+Public+24Q2&file=OmicsExpressionProteinCodingGenesTPMLogp1.csv" \
    -o data/raw/ccle/CCLE_expression_TPM.csv \
    --progress-bar \
    --max-time 600
fi

# ============================================================================
# CCLE Methylation (RRBS TSS ±1kb)
# ============================================================================

echo ""
echo "[2/2] Downloading CCLE Methylation (RRBS TSS ±1kb)..."
echo "This may take 1-3 minutes (~39 MB compressed)"
echo ""
echo "Using CCLE_RRBS_TSS1kb_20181022.txt.gz (RECOMMENDED)"
echo "  - Strongest M → E causal signal"
echo "  - TSS ±1kb captures core promoter methylation"
echo "  - Direct gene-level mapping"
echo ""

# Try the CCLE 2019 release (most stable) - TSS 1kb version
curl -L "https://depmap.org/portal/download/all/?release=CCLE+2019&file=CCLE_RRBS_TSS1kb_20181022.txt.gz" \
  -o data/raw/ccle/CCLE_RRBS_methylation.txt.gz \
  --progress-bar \
  --max-time 600 \
  || echo "Warning: TSS1kb download failed, trying alternative URL..."

# If failed, try alternative format
if [ ! -s data/raw/ccle/CCLE_RRBS_methylation.txt.gz ] || [ $(wc -c < data/raw/ccle/CCLE_RRBS_methylation.txt.gz) -lt 100000 ]; then
  echo "Trying alternative Methylation release..."
  curl -L "https://depmap.org/portal/download/all/?releasename=Methylation+%28RRBS%29&filename=CCLE_RRBS_TSS1kb_20181022.txt.gz" \
    -o data/raw/ccle/CCLE_RRBS_methylation.txt.gz \
    --progress-bar \
    --max-time 600
fi

# Decompress if we got the .gz file
if [ -f data/raw/ccle/CCLE_RRBS_methylation.txt.gz ] && [ $(wc -c < data/raw/ccle/CCLE_RRBS_methylation.txt.gz) -gt 100000 ]; then
  echo "Decompressing methylation data..."
  gunzip -f data/raw/ccle/CCLE_RRBS_methylation.txt.gz
fi

# ============================================================================
# Verification
# ============================================================================

echo ""
echo "=== Download Verification ==="

# Check expression
if [ -f data/raw/ccle/CCLE_expression_TPM.csv ]; then
  EXPR_SIZE=$(ls -lh data/raw/ccle/CCLE_expression_TPM.csv | awk '{print $5}')
  EXPR_LINES=$(wc -l < data/raw/ccle/CCLE_expression_TPM.csv)
  echo "✓ Expression: $EXPR_SIZE ($EXPR_LINES lines)"

  # Check if it's HTML (error page)
  if head -1 data/raw/ccle/CCLE_expression_TPM.csv | grep -q "<!doctype\|<html"; then
    echo "  ⚠️  WARNING: File appears to be HTML (error page), not CSV!"
    echo "  Manual download required - see DEPMAP_REQUIRED_FILES.md"
  elif [ $EXPR_LINES -lt 100 ]; then
    echo "  ⚠️  WARNING: File has too few lines ($EXPR_LINES)"
    echo "  Manual download required - see DEPMAP_REQUIRED_FILES.md"
  else
    echo "  ✓ File looks valid (has $EXPR_LINES lines)"
  fi
else
  echo "✗ Expression: Download failed"
fi

# Check methylation
if [ -f data/raw/ccle/CCLE_RRBS_methylation.txt ]; then
  METH_SIZE=$(ls -lh data/raw/ccle/CCLE_RRBS_methylation.txt | awk '{print $5}')
  METH_LINES=$(wc -l < data/raw/ccle/CCLE_RRBS_methylation.txt)
  echo "✓ Methylation: $METH_SIZE ($METH_LINES lines)"

  # Check if it's HTML
  if head -1 data/raw/ccle/CCLE_RRBS_methylation.txt | grep -q "<!doctype\|<html"; then
    echo "  ⚠️  WARNING: File appears to be HTML (error page)!"
    echo "  Manual download required - see DEPMAP_REQUIRED_FILES.md"
  elif [ $METH_LINES -lt 100 ]; then
    echo "  ⚠️  WARNING: File has too few lines ($METH_LINES)"
    echo "  Manual download required - see DEPMAP_REQUIRED_FILES.md"
  else
    echo "  ✓ File looks valid (has $METH_LINES lines)"
  fi
else
  echo "✗ Methylation: Download failed"
fi

echo ""
echo "=== Next Steps ==="

# Check if both files are valid
EXPR_VALID=false
METH_VALID=false

if [ -f data/raw/ccle/CCLE_expression_TPM.csv ]; then
  if ! head -1 data/raw/ccle/CCLE_expression_TPM.csv | grep -q "<!doctype\|<html"; then
    if [ $(wc -l < data/raw/ccle/CCLE_expression_TPM.csv) -gt 100 ]; then
      EXPR_VALID=true
    fi
  fi
fi

if [ -f data/raw/ccle/CCLE_RRBS_methylation.txt ]; then
  if ! head -1 data/raw/ccle/CCLE_RRBS_methylation.txt | grep -q "<!doctype\|<html"; then
    if [ $(wc -l < data/raw/ccle/CCLE_RRBS_methylation.txt) -gt 100 ]; then
      METH_VALID=true
    fi
  fi
fi

if [ "$EXPR_VALID" = true ] && [ "$METH_VALID" = true ]; then
  echo "✓ SUCCESS: Both files downloaded and verified!"
  echo ""
  echo "You can now proceed with:"
  echo "  python scripts/02_run_inference.py --quick-test"
  echo ""
elif [ "$EXPR_VALID" = false ] || [ "$METH_VALID" = false ]; then
  echo "⚠️  MANUAL DOWNLOAD REQUIRED"
  echo ""
  echo "The automated download encountered API issues."
  echo "Please download manually:"
  echo ""

  if [ "$EXPR_VALID" = false ]; then
    echo "1. Expression Data:"
    echo "   Visit: https://depmap.org/portal/download/"
    echo "   Search: 'OmicsExpression' or 'TPM'"
    echo "   Download: OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    echo "   Save as: data/raw/ccle/CCLE_expression_TPM.csv"
    echo ""
  fi

  if [ "$METH_VALID" = false ]; then
    echo "2. Methylation Data:"
    echo "   Visit: https://depmap.org/portal/download/"
    echo "   Search: 'RRBS' or 'methylation'"
    echo "   Download: CCLE_RRBS_TSS1kb or TSS_CpG_clusters file"
    echo "   Save as: data/raw/ccle/CCLE_RRBS_methylation.txt"
    echo ""
  fi

  echo "See DEPMAP_REQUIRED_FILES.md for detailed instructions"
fi

echo ""
echo "=== Download Complete ==="
