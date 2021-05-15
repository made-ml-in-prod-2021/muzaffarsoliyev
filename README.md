ml_project
==============================
This project is aimed to solve Heart Disease UCI classification problem. It can both train model and predict using existing model. EDA is also included in **notebooks** folder.


### Installation (assuming you are in ml_project/ml_project): ###
- conda activate my-env
- pip install -e .

### Usage (assuming you are in ml_project/ml_project): ###

#### Training ####
python src/train_pipeline.py configs/train_config.yaml

#### Prediction ####
python src/predict.py

#### Tests ####
pytest tests/


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── configs            <- yaml conf files
    │   ├── logging_config.yml 
    │   ├── train_config.yml
    │   └── model_configs
    │       ├── decision_tree_classifier.yaml
    │       └── logistic_regression.yaml
    │
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    │   └── predicted      <- Predictions
    │
    ├── logs               <- Log files
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── outputs            <- Hydra outputs.        
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── model_fit_predict.py    
    │   │
    │   └── enities        <- Dataclasses and functions to parse yaml config files
    │       ├── feature_params.py
    │       ├── model_params.py
    │       ├── split_params.py
    │       ├── train_params.py
    │       └── train_pipeline_params.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
See https://nsls-ii.github.io/scientific-python-cookiecutter/preliminaries.html and https://drivendata.github.io/cookiecutter-data-science/ 
