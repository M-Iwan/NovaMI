![](files/NovaMI.png)

### What is this repository?
A collection of ML/AI code for chemistry applications developed during my PhD. 

Primarily for personal use: function signatures and repository structure may change at any time. Feel free to use any code you find it helpful, but mind that the code written at the beginning of my PhD (February 2024), might be..., bad ¯\_(ツ)_/¯. New code works just fine though!

For convenience, .py files that work and are up-to-date are marked with *, while those unlikely to work (and should be removed) with #. If no symbol is given, the code should work, but won't be compatibile with the rest of the repository.

### Repository Structure
Last updated on version: 0.3.0

```
novami/
├── deprecated/   Old code kept because it might be useful one day
├── environments/   Environments for special-need code (CDDD, Mordred) 
├── novami/
│   ├── api/
│   │   ├── convert.py*   Conversion from IUPAC names to SMILES using OPSIN
│   │   └── resolve.py*   Get SMILES from name using PubChem, CIR, or (WIP) CAS
│   ├── chemistry/
│   │   └── molecule.py   Standardize molecule, embed it in 3D, and dock using SMINA
│   ├── cli/
│   │   ├── CDDD.py*   CLI wrapper for CDDD descriptors
│   │   ├── CIR.py   CLI wrapper of CIR name resolver
│   │   ├── Dock.py  Wrapper for molecule.py
│   │   ├── MolStandardizer.py*   Molecule standardization using RDKit
│   │   ├── Mordred.py*   CLI wrapper for Mordred descriptors
│   │   └── OptunaOptimization.py#   Old code for Optuna optimization.
│   ├── data/
│   │   ├── cluster.py*   Clustering using Butina/Murcko/Connected Components algorithms
│   │   ├── descriptors.py*   Descriptor calculations: ECFP, MACCS, Klek, CDDD, RDKit, Mordred, ChemBERTa, MAPC
│   │   ├── filter.py*   Filter outliers based on molecular parameters
│   │   ├── manager.py*   Main class for managing data during training and inference
│   │   ├── manipulate.py*   Helper functions for checks/data manipulation  
│   │   ├── partition.py*   Partitioning algoriths; convenience wrappers around scikit-learn and cluster.py
│   │   ├── process.py#   Old code to be merged with manipulate or remove
│   │   ├── similarity.py*   Parallel distance matrix / k-neighbors calculations
│   │   └── transform.py*   Main class for normalizing/processing data before training
│   ├── deep/  When I was writing this code, me and God knew how it worked, guess who doesn't anymore?
│   │   ├── dataset.py   String, Graph, and Tensor-based datasets
│   │   ├── model.py   DL models and their individual modules (see MMMTGNN - Multi Modal, Multi Task General Neural Network)
│   │   ├── models.py   Placeholder for cleaned models
│   │   ├── utils.py   Helper functions
│   │   └── vectorizer.py   Graph and String (SMILES, SELFIES, DeepSMILES) vectorizers 
│   ├── io/
│   │   ├── database.py*   Preprocessing of ChEMBL and BindingDB files
│   │   └── file.py*   IO functions for several formats I'm using; works with Pandas/Polars DFs
│   ├── metrics/
│   │   └── modellability.py*   MODI index
│   ├── ml/
│   │   ├── evaluate.py#   Old code for model/ensemble evaluation
│   │   ├── models.py*   Self-contained, sklearn-compatibile models and ensembles; I'm very happy with this one :)
│   │   ├── optimize.py*   Hyperparameter optimization; functions at the top of the file are outdated
│   │   ├── params.py*   Pre-defined parameters for Optuna
│   │   ├── score.py#   Functions for scoring models; Outdated, now included with Unit and Ensemble classes
│   │   ├── select.py#   Sequential feature selection
│   │   └── utils.p*   Helper functions for building Units from just names of models and descriptors
│   ├── nlp/  
│   │   ├── article.py   Article class for retrieving metadata based on DOI/Names
│   │   ├── cluster.py   Latent Dirichlet Allocation for abstract-based clustering
│   │   └── tokenize.py   Word tokenizers
│   ├── standardize/   
│   │   ├── clean.py*   Wrappers around RDKit functions for standaradizing SMILES
│   │   ├── duplicates.py*   Duplicate processing based on Median Absolute Deviation
│   │   ├── filter.py*   Filter based on selected descriptors
│   │   └── validate.py   Check validity of structure
│   └── visualize/
│       ├── ecdf.py*   Emprical Cumulative Distribution Function of molecular inter-distance 
│       ├── embedding.py*   t-SNE and UMAP
│       ├── performance.py   WIP: AU-GOOD framework-related plots
│       ├── predictions.py*   Bunch of plots for assessing model performance; currently only Regression
│       ├── proeprties.py*   Plot and compare molecular properties between datasets
│       └── utils.py*   Helper functions and my custom palette
├── projects/
│   ├── cddd_setup*   Files for setting up CDDD environment anywhere
│   ├── drid   VERY OLD code for the CARBIDE project (https://github.com/M-Iwan/CARBIDE)
│   ├── osmordred_setup   WIP: Corrected Mordred descriptors
│   └── qcg_template*   Template for QCG PilotJob training on bigger scale (one node)
├── temp/   Storage for temporary files
├── tests/   Whatever I'm developing at the moment
├── .gitignore   Ignored Files
├── CHANGELOG.md   List of more important changes between versions
├── code_dev.ipynb   Currently tested/developed additions
├── LICENSE   Self-explanatory
├── README.md   This file!
└── setup.py   Most of required libraries, some day I *might* add specific versions.
```

### Changelog

#### [0.3.0]: 29-01-2026
Public release of repository.
