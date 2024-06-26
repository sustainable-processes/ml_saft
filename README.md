# DL4Thermo

This project aims to build a method for predicting equation of state parameters directly from molecular structures. We use PC-SAFT as the equation of state in this case study.

## Research 

The academic paper can be found [here](https://chemrxiv.org/engage/chemrxiv/article-details/6456371107c3f029374e6608).

### Datasets

We use the following to build a training dataset:

* **[ThermoML Archive](https://trc.nist.gov/ThermoML/)**: A open-source dataset of 50k+ records from several thermodynamic journals spanning from 2003 to 2019. It contains pure component, binary and ternary mixtures.  

* **[Dortmund Databank](http://dortmunddatabank.com/)**: A proprietary dataset that contains a mix of publicly available and proprietary data curated by experts. We have the [2021 VLE databank](http://www.ddbst.com/files/files/ddbsp/2021/Documentation/ReleaseNotes.pdf), which contains data for 41,400 mixtures.   PLEASE NOTE THAT DORTMUND DATABANK CAN ONLY BE USED IN THE COURSE OF THIS PROJECT AND MUST BE DELETED AT ITS TERMINATION.


## Getting started

This project uses `Kedro 0.18.0`, a packaage for managing data science projects. You can read more about Kedro in their [documentation](https://kedro.readthedocs.io) to get started. However, step-by-step instructions for this project are given below.


## Developer setup

First, clone the repository:

```
git clone https://github.com/sustainable-processes/dl4thermo.git
```

We're using [git LFS]((https://docs.gitlab.com/ee/topics/git/lfs/#using-git-lfs)) for large data files, so make sure to set that up. Run the following to get the large files:

```
git lfs fetch origin main
```

Then, make sure to unzip the directories in `data/01_raw`.


### How to install dependencies

We use poetry for dependency management. First, install [poetry](https://python-poetry.org/docs/#installation) and then run:

```
poetry install --with=dev
poe install-pyg
```
This will download all the dependencies.

**Notes for Mac users**:
- Note that if you run on Mac OS, you'll need to have Ventura (13.1 or higher).
- First run this set of commands:
    ```
    arch -arm64 brew install llvm@11
    brew install hdf5
    HDF5_DIR=/opt/homebrew/opt/hdf5 PIP_NO_BINARY="h5py" LLVM_CONFIG="/opt/homebrew/Cellar/llvm@11/11.1.0_3/bin/llvm-config" arch -arm64 poetry install
    ```
  If this commnad fails, make sure to check the version of LLVM that was actually installed (i.e., run ls /opt/homebrew/Cellar/llvm@11/) and replace 11.1.0_3 in the third line above with the correct version.

### Making predictions

Use the `make_predictions.py` command line script to make predictions of PCP-SAFT parameters. The scripts takes in a path to a text file with a SMILES string on each line.

```bash
python make_predictions.py smiles.txt
```

where smiles.txt might be

```
C/C=C(/C)CC
CC(C)CC(C)N
CCCCCCN
CC(C)CC(=O)O
```
    

### How to visualize and run Kedro pipelines

Kedro is centered around [pipelines](https://kedro.readthedocs.io/en/0.18.0/nodes_and_pipelines/pipeline_introduction.html). Pipelines are a series of functions (called nodes) that transform data.

The easiest way to understand pipelines in this repository is using `kedro-viz`. `kedro-viz` is a web app that visualizes all pipelines via a dependency graph, as shown in the example below.

![Kedro Viz example](static/kedro_viz_example.png)

You can run `kedro-viz` using the following command:

```
kedro viz --autoreload
```
Add the autoreload flag to have the visualization automatically refresh on code changes.

You can run all Kedro pipelines in the project using:

```
kedro run
```

However, running all pipelines is usually unnecessary since many of the data processing outcomes are cached via `git LFS`.  Instead, you can run a specific pipeline. For example, here is how to run only the Dortmund database processing pipeline:

```
kedro run --pipeline ddb
```

### Configuration

Kedro relies on a configuration system that can be a bit unintuitive. All configuration is stored inside `conf`. Inside `conf`, you will see the following directories:

- `base`: Used for configuration stored in git
- `local`: Used for configuration only stored on your machine

Each of the above directories has two important configuration filees:

- `catalog.yml`: Describes the [Data Catalog](https://kedro.readthedocs.io/en/0.18.0/data/data_catalog.html), a flexible way of importing and exporting data to/from pipelines. I strongly recommend skimming the [examples](https://kedro.readthedocs.io/en/0.18.0/data/data_catalog.html) in the documentation to get a feel for how the Data Catalog works
- `parameters/**.yml`: The parameters files for each pipeline contain parameters that can be referenced inside the pipeline. Note that parameters are needed for **any node input that does not come from the Data Catalog or a previous node**. To see how to reference parameters, look at the Dortmund data processing pipeline. 


### Rough instructions to reproduce results from our paper

1. Process the Dortmund data. This includes resolving SMILES strings, filtering data, and generating conformers.

  ```bash
  kedro run --pipeline ddb
  kedro run --pipeline ddb_model_prep
  ```

2. (Optional) Generate COSMO-RS pretraining data.

  ```bash
  kedro run --pipeline cosmo
  ```

3. Train model for predicting dipole moments

  ```bash
  kedro run --pipeline train_spk_mu_model
  ```

4. Fit PC-SAFT parameters

  ```bash
  # Regression on Dortmund data
  kedro run --pipeline pcsaft_regression

  # Generate LaTex table with stats about fitting
  kedro run --pipeline pcp_saft_fitting_results_table

  # (Optional) Run regression on COSMO-RS data
  kedro run --pipeline pcsaft_cosmo_regression
  kedro run --pipeline pcp_saft_cosmo_fitting_results_table
  ```

5. Train models

  ```bash
  # Preprocess data
  kedro run --pipeline prepare_regressed_split

  # Train feed-forward nework
  kedro run --pipeline train_ffn_regressed_model

  # Train random forest
  kedro run --pipeline train_sklearn_regressed_model

  # Train D-MPNN model
  kedro run --pipeline train_chemprop_regressed_model

  # Train MPNN model
  kedro run --pipeline train_pyg_regressed_model

  # Pretrain on COSMO-RS and fine-tune on Dortmund
  # Make sure to update the checkpoint_artifact_id under
  # pyg_regressed_pcp_saft_model.train_args in 
  # conf/base/parameters/train_models.yml
  # to the one from the pretrain run on wandb
  kedro run --pipeline train_pyg_pretrain_cosmo_model
  kedro run --pipeline train_pyg_regressed_model
  ```

6. Evaluate models

  Make sure update `results_table` in `conf/base/parameters/results_analysis.yml` with the correct wandb files.

  ```bash
  kedro run --pipeline  results_table
  ```

### How to work with Kedro and notebooks


To use JupyterLab, run: 

```
kedro jupyter lab
```

And if you want to run an IPython session:

```
kedro ipython
```

For both, you will need to run the following cell magic to get the [Kedro global variables](https://kedro.readthedocs.io/en/latest/11_tools_integration/02_ipython.html#load-datacatalog-in-ipython).

```python
%load_ext kedro.extras.extensions.ipython
%reload_kedro
```

You can then load data like this:

```python
df = catalog.load("imported_thermoml")
```

### Merging notebooks

Jupyter notebooks are notoriously difficult to merge. Use [nbdime](https://nbdime.readthedocs.io/en/latest/vcs.html#id4) to help with this, which is installed as part of the development dependencies. If you get a failed merge, do the following:

1. Register nbdime as a merge tool:
    ```bash
    git-nbmergetool config --enable 
    ```

2. View the nbdime merge web interface on the file that failed to automerge

    ```bash
    git mergetool --tool=nbdime [<file>…​]
    ```


 ## Citations

- Chemprop
- The `chem_utils` code comes from [Kyle Swanson](https://github.com/swansonk14/chem_utils/)

