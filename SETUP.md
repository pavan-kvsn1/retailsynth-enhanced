# RetailSynth Enhanced - Setup Guide

This guide will help you set up your development environment for the RetailSynth Enhanced project.

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- **Git**
- **wget** or **curl** (for downloading Dunnhumby data)
- **unzip** utility
- **Optional**: CUDA-capable GPU for JAX acceleration

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/[your-org]/retailsynth-enhanced.git
cd retailsynth-enhanced
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n retailsynth python=3.11
conda activate retailsynth
```

### 3. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (for contributors)
pip install -r requirements-dev.txt
```

### 4. Download Dunnhumby Dataset

```bash
# Make script executable
chmod +x scripts/download_dunnhumby.sh

# Run download script
./scripts/download_dunnhumby.sh
```

This will:
- Download the Dunnhumby Complete Journey dataset (~500MB)
- Extract all CSV files
- Verify file integrity
- Display dataset summary

**Expected files:**
- `product.csv` - 92,000+ products
- `transaction_data.csv` - 84M+ transactions
- `hh_demographic.csv` - 2,500 households
- `causal_data.csv` - Promotional data
- Additional campaign and coupon files

### 5. Verify Installation

```bash
# Run tests
pytest tests/unit/ -v

# Check JAX installation
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"

# Check data
python -c "import pandas as pd; print(f'Products: {len(pd.read_csv(\"data/raw/dunnhumby/product.csv\"))}')"
```

## Development Setup

### Pre-commit Hooks

Install pre-commit hooks for automatic code formatting and linting:

```bash
pre-commit install
```

This will run the following checks before each commit:
- **Black** - Code formatting
- **isort** - Import sorting
- **Ruff** - Linting
- **MyPy** - Type checking

### IDE Configuration

#### VS Code

Recommended extensions:
- Python (Microsoft)
- Pylance
- Ruff
- Black Formatter
- GitLens

Workspace settings (`.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false
}
```

#### PyCharm

1. Set Python interpreter: `Settings → Project → Python Interpreter`
2. Enable Black: `Settings → Tools → Black`
3. Enable pytest: `Settings → Tools → Python Integrated Tools → Testing`

## Project Structure

```
retailsynth-enhanced/
├── src/retailsynth/          # Main package
│   ├── catalog/              # Product catalog (Sprint 1.1)
│   ├── engines/              # Behavioral engines
│   ├── generators/           # Data generators
│   ├── calibration/          # Calibration system
│   └── validation/           # Validation tests
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── validation/           # Validation tests
├── scripts/                  # Utility scripts
├── configs/                  # Configuration files
├── data/                     # Data directory (gitignored)
│   ├── raw/dunnhumby/        # Dunnhumby dataset
│   └── processed/            # Processed data
└── docs/                     # Documentation
```

## Configuration

### Default Configuration

The default configuration is in `configs/default.yaml`:

```yaml
# Number of entities
n_customers: 10000
n_products: 20000  # From Dunnhumby catalog
n_stores: 50
n_weeks: 104

# Paths
product_catalog_path: "data/processed/product_catalog/product_catalog_20k.parquet"
category_hierarchy_path: "data/processed/product_catalog/category_hierarchy.json"

# Random seed for reproducibility
random_seed: 42
```

### Custom Configuration

Create a custom config file:

```bash
cp configs/default.yaml configs/my_config.yaml
# Edit my_config.yaml
```

Use it:

```python
from retailsynth import RetailSynthGenerator
from retailsynth.config import load_config

config = load_config('configs/my_config.yaml')
generator = RetailSynthGenerator(config)
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/              # Unit tests only
pytest tests/integration/       # Integration tests only
pytest tests/validation/        # Validation tests only

# Run with coverage
pytest --cov=src/retailsynth --cov-report=html

# Run in parallel (faster)
pytest -n auto

# Run specific test file
pytest tests/unit/test_basket_composer.py -v
```

## Building Documentation

```bash
cd docs
make html

# View documentation
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
```

## Common Issues

### Issue: JAX not detecting GPU

**Solution:**
```bash
# Uninstall CPU version
pip uninstall jax jaxlib

# Install GPU version
pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Issue: Download script fails

**Solution:**
- Check internet connection
- Verify wget/curl is installed
- Try manual download from: https://www.dunnhumby.com/source-files/

### Issue: Import errors

**Solution:**
```bash
# Ensure package is installed in editable mode
pip install -e .
```

### Issue: Tests fail with "No module named 'retailsynth'"

**Solution:**
```bash
# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or install package
pip install -e .
```

## Next Steps

Once setup is complete, proceed to:

1. **Sprint 1.1**: Product Catalog Alignment
   ```bash
   python scripts/build_product_catalog.py
   ```

2. **Sprint 1.2**: Price Elasticity Learning
   ```bash
   python scripts/learn_price_elasticity.py
   ```

3. **Generate Synthetic Data**
   ```bash
   python scripts/generate_synthetic.py --config configs/calibrated.yaml
   ```

## Getting Help

- **Documentation**: See `docs/` directory
- **Issues**: Create a GitHub issue using templates
- **Discussions**: Use GitHub Discussions for questions

## Contributing

See `CONTRIBUTING.md` for contribution guidelines.

## Phase 0 Checklist

- [x] Environment setup
- [x] CI/CD workflows configured
- [x] Issue templates created
- [x] Download script ready
- [ ] Dunnhumby data downloaded
- [ ] Tests passing
- [ ] Ready for Sprint 1.1!

---

**Ready to start development! 🚀**
