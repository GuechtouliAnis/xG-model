# xG Model

## Repository Structure

```
xG_Project/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── __init__.py
│   ├── constants.py
│   ├── get_data.py
│   ├── xG_constants.py
│   ├── xG_preprocessing.py
│   ├── xG_models.py
│   ├── xG_evaluation.py
│   └── xG_visualization.py
│
├── notebooks/
│   ├── 1_Data_Preprocessing_and_EDA.ipynb
│   ├── 2_Model_Training_and_Evaluation.ipynb
│   ├── 3_Tactical_Analysis_and_Insights.ipynb
│   ├── data_splitter.ipynb
│   └── importing_data.ipynb
│
├── models/
│   ├── best_model.pkl
│   ├── model_comparison.csv
│   └── README.md
│
├── outputs/
│   ├── figures/
│   ├── reports/
│   └── exports/
│
├── docs/
│   ├── Open Data 360 Frames v1.0.0.pdf
│   ├── Open Data Competitions v2.0.0.pdf
│   ├── Open Data Events v4.0.0.pdf
│   ├── Open Data Lineups v2.0.0
│   ├── Open Data Matches v3.0.0.pdf
│   └── StatsBomb Open Data Specification v1.1.pdf
│
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py
    └── test_models.py
```


## Description

## Feature Selection

### Dropped Tables

### Tables Used

#### Events

#### Frames

The `frames` table contains the positions of each player during each event for competitions covered by the StatsBomb 360 data. The competitions included are:

- `1. Bundesliga 2023/2024 - Bayer Leverkusen`
- `UEFA Euro 2020`
- `FIFA World Cup 2022`
- `UEFA Euro 2024`

This table was processed in the <a href='cleaning_frames.ipynb'>`cleaning_frames.ipynb`</a> notebook by splitting the location of events to facilitate easier access and use of the data.

## References