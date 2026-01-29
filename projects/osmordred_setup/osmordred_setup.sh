# Clone and enter the repository
git clone https://github.com/osmoai/osmordred.git
cd osmordred

# Setup the environment
conda create -y -n osmordred -c conda-forge python=3.11 boost==1.82.0 eigen lapack ninja python-build rdkit==2023.9.3
conda activate osmordred

# Move some files that are in wrong place
cp ./skbuild/setup.py .
cp ./skbuild/build.sh .

# Make changes to the setup.py files
sed -i 's/from skbuild import setup//g' setup.py
sed -i 's/from setuptools import find_packages/import setuptools/g' setup.py
sed -i 's/^setup/setuptools.setup/g' setup.py
sed -i 's/\["osmordred"]/setuptools.find_packages()/g' setup.py

# Build
python -m build

# Install the library
pip install dist/osmordred-0.2.0*.whl