# xG Model

## Repository Structure

```
-- xG-Model
    |-- Documentation
    |   |-- Open Data 360 Frames v1.0.0.pdf
    |   |-- Open Data Competitions v2.0.0.pdf
    |   |-- Open Data Events v4.0.0.pdf
    |   |-- Open Data Lineups v2.0.0.pdf
    |   |-- Open Data Matches v2.0.0.pdf
    |   |-- StatsBomb Open Data Specification v1.1.pdf
    |-- examples
    |-- notebooks
    |   |-- cleaning_frames.ipynb
    |   |-- data_splitter.ipynb
    |   |-- statsbomb_data.ipynb
    |   |-- xG8.ipynb
    |-- reports
    |-- scripts
    |-- src
    |   |-- __init__.py
    |   |-- main.py
    |   |-- SB_collector.py
    |   |-- SB_data_plitter.py
    |   |-- xG_evaluation.py
    |   |-- xG_models.py
    |   |-- xG_preprocessing.py
    |-- tests
    |-- .gitignore
    |-- CHANGELOG.md
    |-- LICENSE
    |-- README.md
    |-- requirements.txt
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