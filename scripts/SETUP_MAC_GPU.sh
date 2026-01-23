#!/bin/bash
# Setup script for running polymer notebook with Mac M5 GPU

set -e  # Exit on error

echo "=================================="
echo "Mac M5 GPU Setup for Polymer Analysis"
echo "=================================="

# Step 1: Activate virtual environment
echo -e "\n[1/5] Activating virtual environment..."
source polymer_env/bin/activate

# Step 2: Upgrade pip
echo -e "\n[2/5] Upgrading pip..."
pip install --upgrade pip

# Step 3: Install all required packages
echo -e "\n[3/5] Installing required packages (this may take 5-10 minutes)..."
pip install jupyter notebook ipykernel nbconvert

# Core scientific packages
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm joblib

# Chemistry packages
pip install rdkit

# Machine learning packages (with MPS support for M5)
pip install torch torchvision torchaudio

# Transformer packages
pip install transformers sentencepiece

# Dimensionality reduction
pip install umap-learn

# Step 4: Install IPython kernel for this environment
echo -e "\n[4/5] Setting up Jupyter kernel..."
python -m ipykernel install --user --name=polymer_env --display-name="Python (Polymer M5 GPU)"

# Step 5: Verify installations
echo -e "\n[5/5] Verifying installations..."
python -c "import torch; print('✓ PyTorch:', torch.__version__)"
python -c "import torch; print('✓ MPS Available:', torch.backends.mps.is_available())"
python -c "import torch; print('✓ MPS Built:', torch.backends.mps.is_built())"
python -c "import rdkit; print('✓ RDKit:', rdkit.__version__)"
python -c "import transformers; print('✓ Transformers:', transformers.__version__)"
python -c "import sklearn; print('✓ Scikit-learn:', sklearn.__version__)"
python -c "import umap; print('✓ UMAP: installed')"

echo -e "\n=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. In VS Code, select kernel: 'Python (Polymer M5 GPU)'"
echo "2. Run the notebook cells"
echo "3. GPU will be used automatically for transformer model!"
echo ""
echo "To activate this environment in terminal:"
echo "  source polymer_env/bin/activate"
echo ""
