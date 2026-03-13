# Forecast_Evaluation
Official repository for the paper: A Survey and Benchmark for Household Electricity Forecasting: From Statistical to Foundation Models

# Time Series Forecasting Models Overview

This repository benchmarks **30 time-series forecasting models** spanning **six major methodological classes**:

1. Statistical & Machine Learning  
2. MLP-based Neural Networks  
3. Recurrent Neural Networks  
4. Convolutional Neural Networks  
5. Transformer-based Models  
6. Time Series Foundation Models  

These models represent the evolution of forecasting techniques from **classical statistical approaches to large-scale pre-trained time-series foundation models**.

---

# Model Categories

| Class | Number of Models |
|------|------|
| Statistical & Machine Learning | 5 |
| MLP-based Models | 5 |
| Recurrent Models | 5 |
| Convolutional Models | 5 |
| Transformer Models | 5 |
| Time Series Foundation Models | 5 |

---

# 1. Statistical & Machine Learning Models

These models rely on classical statistical assumptions or traditional machine learning techniques. They are typically **computationally efficient**, easy to interpret, and often serve as **strong baseline models**.

| Model | Description |
|------|-------------|
| [AutoARIMA](https://github.com/alkaline-ml/pmdarima) | Automated implementation of the ARIMA model that selects optimal parameters using statistical tests and information criteria. |
| [KNN Regression](https://github.com/scikit-learn/scikit-learn) | Instance-based regression model that predicts values based on the most similar historical observations. |
| [LightGBM](https://github.com/microsoft/LightGBM) | Gradient boosting decision tree framework designed for efficiency and scalability with strong performance on structured data. |
| [Naive Drift](https://github.com/robjhyndman/forecast) | Baseline model assuming a linear drift between the first and last observations. |
| [Naive Moving Average](https://github.com/robjhyndman/forecast) | Forecasts future values by averaging recent historical observations. |

**Characteristics**

- Simple and interpretable  
- Low computational requirements  
- Often strong baseline performance

---

# 2. MLP-Based Models

MLP-based models use **feedforward neural networks** to map historical input windows directly to future predictions. They capture nonlinear relationships but lack explicit sequential memory.

| Model | Description |
|------|-------------|
| [DeepNPTS](https://github.com/awslabs/gluonts) | Non-parametric probabilistic forecasting model that samples predictions directly from historical distributions. |
| [N-BEATS](https://github.com/ServiceNow/N-BEATS) | Deep residual architecture that decomposes time series into interpretable trend and seasonal components. |
| [NHITS](https://github.com/Nixtla/neuralforecast) | Hierarchical interpolation model extending N-BEATS to capture multi-scale temporal patterns. |
| [NLinear](https://github.com/cure-lab/LTSF-Linear) | Lightweight linear forecasting model designed for long-term forecasting tasks. |
| [TiDE](https://github.com/google-research/google-research/tree/master/tide) | Encoder–decoder architecture using dense layers to process historical values and covariates efficiently. |

**Characteristics**

- Efficient training  
- Strong nonlinear modeling capability  
- Limited temporal memory

---

# 3. Recurrent Neural Network Models

Recurrent architectures explicitly model **temporal dependencies** using sequential structures such as LSTM, GRU, or state-space mechanisms.

| Model | Description |
|------|-------------|
| [DeepAR](https://github.com/awslabs/gluonts) | Probabilistic autoregressive RNN model designed for large-scale time series forecasting. |
| [DeepFactor](https://github.com/awslabs/gluonts) | Combines global deep neural networks with local probabilistic models. |
| [MQ-RNN](https://github.com/awslabs/gluonts) | Recurrent architecture designed for multi-horizon quantile forecasting. |
| [Mamba](https://github.com/state-spaces/mamba) | State-space sequence model designed for efficient long-sequence modeling. |
| [Temporal Fusion Transformer (TFT)](https://github.com/google-research/google-research/tree/master/tft) | Hybrid architecture combining recurrent layers and attention mechanisms for interpretable forecasting. |

**Characteristics**

- Effective at capturing temporal dependencies  
- Strong sequential modeling capabilities  
- Higher computational cost compared to simple neural models

---

# 4. Convolutional Neural Network Models

CNN-based models apply **causal and dilated convolutions** to capture temporal patterns efficiently while allowing parallel computation.

| Model | Description |
|------|-------------|
| [TCN](https://github.com/locuslab/TCN) | Temporal convolutional network using dilated causal convolutions to capture long-range dependencies. |
| [BiTCN](https://github.com/Nixtla/neuralforecast) | Bidirectional TCN architecture capturing forward and backward temporal information. |
| [TimesNet](https://github.com/thuml/TimesNet) | Converts time series into 2D representations to model periodic patterns using convolutional operations. |
| [WaveNet](https://github.com/ibab/tensorflow-wavenet) | Dilated convolution architecture originally developed for audio generation and adapted for forecasting tasks. |
| [MQ-CNN](https://github.com/awslabs/gluonts) | CNN-based architecture designed for multi-horizon quantile forecasting. |

**Characteristics**

- Efficient parallel computation  
- Strong local pattern detection  
- Good scalability for large datasets

---

# 5. Transformer-Based Models

Transformer architectures leverage **self-attention mechanisms** to model long-range dependencies without recurrence.

| Model | Description |
|------|-------------|
| [Informer](https://github.com/zhouhaoyi/Informer2020) | Efficient transformer architecture using ProbSparse attention for long sequence forecasting. |
| [PatchTST](https://github.com/yuqinie98/PatchTST) | Transformer model using patch-based tokenization to improve efficiency and performance. |
| [iTransformer](https://github.com/thuml/iTransformer) | Transformer variant that treats variables as tokens to improve multivariate correlation modeling. |
| [Vanilla Transformer](https://github.com/huggingface/transformers) | Standard transformer architecture adapted for time series forecasting tasks. |
| [TimeXer](https://github.com/thuml/TimeXer) | Transformer architecture designed to integrate exogenous variables through cross-attention mechanisms. |

**Characteristics**

- Excellent long-range dependency modeling  
- Flexible attention mechanisms  
- High computational requirements for long sequences

---

# 6. Time Series Foundation Models

Time Series Foundation Models (TSFMs) are **large-scale pre-trained models trained on massive datasets**, enabling **zero-shot forecasting** without task-specific training.

| Model | Description |
|------|-------------|
| [TimeGPT](https://github.com/Nixtla/nixtla) | Transformer-based foundation model trained on large time-series datasets for zero-shot forecasting. |
| [TimesFM](https://github.com/google-research/timesfm) | Patch-based transformer architecture optimized for scalable long-horizon forecasting. |
| [MOIRAI](https://github.com/SalesforceAIResearch/moirai) | Universal forecasting transformer supporting multivariate time series and long prediction horizons. |
| [Chronos](https://github.com/amazon-science/chronos-forecasting) | Token-based transformer model trained on quantized time-series data. |
| [Timer-XL](https://github.com/thuml/Large-Time-Series-Model) | Long-context transformer model designed for capturing both intra-series and inter-series dependencies. |

**Characteristics**

- Pre-trained on massive datasets  
- Capable of zero-shot or few-shot forecasting  
- High computational requirements

---

# Summary

The benchmark evaluates **30 forecasting models** spanning classical statistical approaches, deep learning architectures, and modern foundation models.

Key observations:

- Classical machine learning models such as **LightGBM** remain highly competitive.
- Deep learning models such as **DeepNPTS** and **DeepAR** show strong performance across tasks.
- Foundation models such as **Timer-XL** demonstrate promising **zero-shot forecasting capabilities**.
- Foundation models often produce **smoother forecasts without task-specific training**.

---

# Citation

If you use this benchmark or repository in your research, please cite the corresponding paper.