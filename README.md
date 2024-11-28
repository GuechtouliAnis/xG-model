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
    |-- .gitignore
    |-- read_data.ipynb
    |-- README.md
    |-- requirements.txt
    |-- SB_Data.ipynb
```


## Description

The `Documentation` folder contains a set of files that explain the features found in the dataset. These documents provide detailed information about the data structure and the meaning of various fields in the dataset. The `Documentation` files were obtained from the <a href='https://github.com/statsbomb/statsbombpy'>statsbombpy GitHub repository</a>, specifically from the `docs` directory.

- The <a href='SB_Data.ipynb'>`SB_Data.ipynb`</a> file contains the code to read the data from the StatsBomb dataset.
- The <a href='read_data.ipynb'>`read_data.ipynb`</a> file contains code to read the data from a local Hadoop server into Spark through PySpark.
- The <a href='requirements.txt'>`requirements.txt`</a> file lists the libraries required to run the code.

All the data used in this project is freely available on the <a href='https://statsbomb.com/what-we-do/hub/free-data/'>StatsBomb website</a>.

## Feature Selection

Given the large number of columns in each table, not all data is relevant for building an xG model. Therefore, careful feature selection is crucial to ensure that only the most pertinent data is used.

### Dropped Tables

The following tables have been excluded because they contain data not useful for the xG model:

- `matches`
- `lineups`
- `competitions`

### Tables Used

#### Events

The `events` table contains data about each event during football matches for the data published by StatsBomb. It is important to note that not every event has detailed information about the positions of all players during the event. The data is limited to the *360 Data*, which is freely available for three competitions, as detailed below.

#### Frames

The `frames` table contains the positions of each player during each event for competitions covered by the StatsBomb 360 data. The competitions included are:

- `1. Bundesliga 2023/2024 - Bayer Leverkusen`
- `UEFA Euro 2020`
- `FIFA World Cup 2022`
- `UEFA Euro 2024`

This table was processed in the <a href='cleaning_frames.ipynb'>`cleaning_frames.ipynb`</a> notebook by splitting the location of events to facilitate easier access and use of the data.

## References