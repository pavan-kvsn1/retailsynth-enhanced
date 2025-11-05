#!/bin/bash
# Download and setup Dunnhumby Complete Journey dataset
# 
# The Dunnhumby Complete Journey dataset is available from:
# https://www.dunnhumby.com/source-files/
# 
# This script automates the download and extraction process.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="data/raw/dunnhumby"
DOWNLOAD_URL="https://www.dunnhumby.com/wp-content/uploads/2020/09/dunnhumby_The-Complete-Journey.zip"
ZIP_FILE="dunnhumby_complete_journey.zip"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Dunnhumby Complete Journey Dataset Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if data directory exists
if [ -d "$DATA_DIR" ] && [ "$(ls -A $DATA_DIR)" ]; then
    echo -e "${YELLOW}Warning: Data directory already exists and is not empty.${NC}"
    read -p "Do you want to re-download? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Skipping download. Using existing data.${NC}"
        exit 0
    fi
    echo -e "${YELLOW}Removing existing data...${NC}"
    rm -rf "$DATA_DIR"
fi

# Create data directory
echo -e "${GREEN}Creating data directory: $DATA_DIR${NC}"
mkdir -p "$DATA_DIR"

# Download dataset
echo -e "${GREEN}Downloading Dunnhumby Complete Journey dataset...${NC}"
echo -e "${YELLOW}Note: This is a large file (~500MB) and may take several minutes.${NC}"
echo ""

if command -v wget &> /dev/null; then
    wget -O "$DATA_DIR/$ZIP_FILE" "$DOWNLOAD_URL" --progress=bar:force 2>&1
elif command -v curl &> /dev/null; then
    curl -L -o "$DATA_DIR/$ZIP_FILE" "$DOWNLOAD_URL" --progress-bar
else
    echo -e "${RED}Error: Neither wget nor curl is installed.${NC}"
    echo -e "${YELLOW}Please install wget or curl and try again.${NC}"
    exit 1
fi

# Check if download was successful
if [ ! -f "$DATA_DIR/$ZIP_FILE" ]; then
    echo -e "${RED}Error: Download failed. File not found.${NC}"
    exit 1
fi

# Extract dataset
echo ""
echo -e "${GREEN}Extracting dataset...${NC}"
unzip -q "$DATA_DIR/$ZIP_FILE" -d "$DATA_DIR"

# Remove zip file to save space
echo -e "${GREEN}Cleaning up...${NC}"
rm "$DATA_DIR/$ZIP_FILE"

# Verify expected files exist
echo ""
echo -e "${GREEN}Verifying dataset files...${NC}"

EXPECTED_FILES=(
    "product.csv"
    "transaction_data.csv"
    "hh_demographic.csv"
    "causal_data.csv"
    "coupon.csv"
    "coupon_redempt.csv"
    "campaign_desc.csv"
    "campaign_table.csv"
)

MISSING_FILES=()
for file in "${EXPECTED_FILES[@]}"; do
    # Check in DATA_DIR and subdirectories
    if ! find "$DATA_DIR" -name "$file" -type f | grep -q .; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All expected files found!${NC}"
else
    echo -e "${YELLOW}Warning: Some expected files are missing:${NC}"
    for file in "${MISSING_FILES[@]}"; do
        echo -e "  ${RED}✗${NC} $file"
    done
fi

# Move files to root of data directory if they're in a subdirectory
echo ""
echo -e "${GREEN}Organizing files...${NC}"
find "$DATA_DIR" -name "*.csv" -type f -exec mv {} "$DATA_DIR/" \; 2>/dev/null || true

# Remove any empty subdirectories
find "$DATA_DIR" -type d -empty -delete 2>/dev/null || true

# Display dataset summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Dataset Summary${NC}"
echo -e "${GREEN}========================================${NC}"

if [ -f "$DATA_DIR/product.csv" ]; then
    PRODUCT_COUNT=$(tail -n +2 "$DATA_DIR/product.csv" | wc -l | tr -d ' ')
    echo -e "Products: ${GREEN}$PRODUCT_COUNT${NC}"
fi

if [ -f "$DATA_DIR/transaction_data.csv" ]; then
    TRANSACTION_COUNT=$(tail -n +2 "$DATA_DIR/transaction_data.csv" | wc -l | tr -d ' ')
    echo -e "Transactions: ${GREEN}$TRANSACTION_COUNT${NC}"
fi

if [ -f "$DATA_DIR/hh_demographic.csv" ]; then
    HOUSEHOLD_COUNT=$(tail -n +2 "$DATA_DIR/hh_demographic.csv" | wc -l | tr -d ' ')
    echo -e "Households: ${GREEN}$HOUSEHOLD_COUNT${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Data location: ${GREEN}$DATA_DIR${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Run: ${GREEN}python scripts/build_product_catalog.py${NC}"
echo -e "2. Run: ${GREEN}python scripts/learn_price_elasticity.py${NC}"
echo -e "3. Start Sprint 1.1: Product Catalog Alignment"
echo ""

# Create a marker file to indicate successful download
touch "$DATA_DIR/.download_complete"

exit 0