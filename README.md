# A Survey and Benchmark for Household Electricity Forecasting: From Statistical to Foundation Models

Accurate energy forecasting is a fundamental requirement for modern power systems. Reliable predictions of electricity consumption and generation enable grid operators, utilities, and energy management systems to improve operational efficiency, reduce costs, and support the integration of distributed energy resources such as photovoltaic (PV) systems, electric vehicles, and battery energy storage systems (BESS). As energy systems become increasingly decentralized and data-driven, forecasting methods must also evolve to handle complex temporal patterns, heterogeneous data sources, and varying levels of data availability. 

Over the past decade, forecasting methodologies have undergone significant transformation. Early approaches were dominated by classical statistical models and traditional machine learning algorithms, which are often interpretable and computationally efficient. More recently, deep learning architectures such as recurrent neural networks, convolutional networks, and transformers have demonstrated strong performance by capturing complex nonlinear and long-range temporal dependencies in time-series data. The latest development in this progression is the emergence of **time series foundation models (TSFMs)**, large pre-trained models capable of performing forecasting tasks using zero-shot inference without task-specific training. 

This work presents a **comprehensive survey and benchmarking study of 30 representative forecasting models**, covering the full spectrum of forecasting approaches, from classical statistical methods to recent TSFMs. To ensure a fair and consistent evaluation, all models are benchmarked within a unified experimental framework. The evaluation focuses on **three important household energy forecasting tasks**:

- **Electricity load forecasting**
- **Solar photovoltaic (PV) generation forecasting**
- **Battery energy storage system (BESS) operation forecasting**

These tasks collectively capture the dynamics of distributed energy systems and allow the assessment of forecasting models across both consumption and generation scenarios. The benchmarking pipeline includes a standardized data processing workflow consisting of interpolation for missing values, outlier detection using statistical methods, normalization, and resampling at different temporal resolutions. All models are evaluated using their **default configurations**, without extensive hyperparameter tuning, in order to assess their **out-of-the-box performance** and ensure comparability across architectures.

The main objectives of this work are therefore to:

- Provide a **comprehensive survey of forecasting models** ranging from classical statistical to modern foundation models
- Establish a **unified benchmarking framework** for fair comparison across different forecasting paradigms
- Evaluate model performance across **multiple real-world energy forecasting tasks**
- Investigate the **zero-shot capabilities and limitations of TSFMs**

![Survey method overview](figures/survey_method.png)

**Highlights**
- 30 models across 6 classes with consistent evaluation pipelines
- Multiple household energy datasets (PV, battery, and load)
- Reproducible runs with per-model logging, plots, and CSV outputs

## Quick Start

### Create environment and install dependencies

This repository does not ship a requirements file yet. A minimal install that matches the code paths is:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install pandas numpy scipy scikit-learn lightgbm pmdarima matplotlib seaborn
python -m pip install torch neuralforecast gluonts mxnet transformers
python -m pip install nixtla timesfm chronos-forecasting mamba-ssm uni2ts
```

Notes:
- Some foundation models require GPU for practical runtimes.
- TimeGPT requires an API key (see below).

### Run a model class script

Each script runs all models in that class and stores results under [results](results).

```bash
python models/models_statsml.py
python models/models_neuralforecast.py
python models/models_gluonts.py
python models/model_mamba.py
python models/model_naivedrift.py
python models/fmodel_timegpt.py
python models/fmodel_timesfm.py
python models/fmodel_moirai.py
python models/fmodel_chronos.py
python models/fmodel_timerxl.py
```

To switch datasets, edit the `selected_dataset` value in the relevant script.

## Datasets

Datasets are loaded via [models/dataset_config.py](models/dataset_config.py). File locations used by the loaders:

- Belgium PV and battery datasets: [data/belgium_dataset](data/belgium_dataset)
- Germany WPUQ (SFH19): [data/germany_wpuq_dataset/SFH19_2023_2024_15min_3_month.csv](data/germany_wpuq_dataset/SFH19_2023_2024_15min_3_month.csv)
- London smart meter dataset: [data/london_dataset/LCL_london_consumption_2013.csv](data/london_dataset/LCL_london_consumption_2013.csv)
- Zonnedael dataset: [data/zonnedael_dataset/liander_zonnedael_2013_original.csv](data/zonnedael_dataset/liander_zonnedael_2013_original.csv)

## Model Catalog (30 Models)

Each model lists its primary library and a clickable GitHub repository. Scripts are linked for direct execution.

![Forecast models timeline](figures/forcast_models_timeline.png)

### 1) Statistical and Machine Learning (5)

| Model | Library | Repository | Script |
|------|---------|------------|--------|
| AutoARIMA | pmdarima | https://github.com/alkaline-ml/pmdarima | [models/models_statsml.py](models/models_statsml.py) |
| KNN Regression | scikit-learn | https://github.com/scikit-learn/scikit-learn | [models/models_statsml.py](models/models_statsml.py) |
| LightGBM | LightGBM | https://github.com/microsoft/LightGBM | [models/models_statsml.py](models/models_statsml.py) |
| Naive Drift | custom (numpy, pandas) | https://github.com/numpy/numpy | [models/model_naivedrift.py](models/model_naivedrift.py) |
| Naive Moving Average | custom (numpy, pandas) | https://github.com/pandas-dev/pandas | [models/models_statsml.py](models/models_statsml.py) |

### 2) MLP-based Models (5)

| Model | Library | Repository | Script |
|------|---------|------------|--------|
| DeepNPTS | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| N-BEATS | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| NHITS | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| NLinear | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| TiDE | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |

### 3) Recurrent Neural Networks (5)

| Model | Library | Repository | Script |
|------|---------|------------|--------|
| DeepAR | GluonTS | https://github.com/awslabs/gluonts | [models/models_gluonts.py](models/models_gluonts.py) |
| DeepFactor | GluonTS | https://github.com/awslabs/gluonts | [models/models_gluonts.py](models/models_gluonts.py) |
| MQ-RNN | GluonTS | https://github.com/awslabs/gluonts | [models/models_gluonts.py](models/models_gluonts.py) |
| Mamba | mamba-ssm | https://github.com/state-spaces/mamba | [models/model_mamba.py](models/model_mamba.py) |
| Temporal Fusion Transformer (TFT) | GluonTS | https://github.com/awslabs/gluonts | [models/models_gluonts.py](models/models_gluonts.py) |

### 4) Convolutional Neural Networks (5)

| Model | Library | Repository | Script |
|------|---------|------------|--------|
| TCN | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| BiTCN | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| TimesNet | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| WaveNet | GluonTS | https://github.com/awslabs/gluonts | [models/models_gluonts.py](models/models_gluonts.py) |
| MQ-CNN | GluonTS | https://github.com/awslabs/gluonts | [models/models_gluonts.py](models/models_gluonts.py) |

### 5) Transformer-based Models (5)

| Model | Library | Repository | Script |
|------|---------|------------|--------|
| Informer | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| PatchTST | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| iTransformer | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| Vanilla Transformer | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| TimeXer | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |

### 6) Time Series Foundation Models (5)

| Model | Library | Repository | Script |
|------|---------|------------|--------|
| TimeGPT | nixtla | https://github.com/Nixtla/nixtla | [models/fmodel_timegpt.py](models/fmodel_timegpt.py) |
| TimesFM | timesfm | https://github.com/google-research/timesfm | [models/fmodel_timesfm.py](models/fmodel_timesfm.py) |
| MOIRAI | uni2ts | https://github.com/SalesforceAIResearch/uni2ts | [models/fmodel_moirai.py](models/fmodel_moirai.py) |
| Chronos | chronos-forecasting | https://github.com/amazon-science/chronos-forecasting | [models/fmodel_chronos.py](models/fmodel_chronos.py) |
| Timer-XL | transformers | https://github.com/huggingface/transformers | [models/fmodel_timerxl.py](models/fmodel_timerxl.py) |

## Model-specific Notes

- TimeGPT requires a valid API key in [models/fmodel_timegpt.py](models/fmodel_timegpt.py).
- Timer-XL uses Hugging Face model IDs. Edit the `model_configs` mapping in [models/fmodel_timerxl.py](models/fmodel_timerxl.py) to switch models.
- Chronos, TimesFM, and MOIRAI may download weights on first run.

## Results

All model outputs are saved under [results](results) with per-run CSVs, plots, and metrics summaries. The plotting and metrics utilities are in [models/utils.py](models/utils.py) and [models/plots.py](models/plots.py).

## Citation

If you use this benchmark or repository in your research, please cite the corresponding paper.