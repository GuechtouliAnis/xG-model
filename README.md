# xG Model

## Repository Structure

```
xG_Project/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── 1_importing_data.ipynb
├── 2_splitting_data.ipynb
├── 3_preprocessing.ipynb
├── 4_exploratory_data_analysis.ipynb
├── 5_model_training.ipynb
├── selected_features.md
│
├── data/
│   ├── preprocessed/
│   ├── raw/
│   └── split_data/
│
├── src/
│   ├── __init__.py
│   ├── constants.py
│   ├── get_data.py
│   ├── xG_constants.py
│   ├── xG_evaluation.py
│   ├── xG_models.py
│   ├── xG_preprocessing.py
│   └── xG_visualization.py
│
└── docs/
    ├── Open Data 360 Frames v1.0.0.pdf
    ├── Open Data Competitions v2.0.0.pdf
    ├── Open Data Events v4.0.0.pdf
    ├── Open Data Lineups v2.0.0
    ├── Open Data Matches v3.0.0.pdf
    └── StatsBomb Open Data Specification v1.1.pdf
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

## References