# xG Model

### Repository Structure

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

### Description

The `Documentation` directory contains a set of files that explain the features found in the data.
These documents provide detailed information about the data structure and the meaning of various fields used in the dataset.
The `Documentation` files were obtained from the <a href='https://github.com/statsbomb/statsbombpy'>statsbombpy GitHub repository</a> in the `docs` directory.

The `SB_Data.ipynb` file contains the code to read the data from the StatsBomb dataset.

The `read_data.ipynb` file contains the code to read the data from `Hadoop` localhost into `Spark` through `PySpark`.

The `requirements.txt` file contains the list of libraries required to run the code.

All the data that has been used in this project is freely available on the <a href='https://statsbomb.com/what-we-do/hub/free-data/'>Statsbomb website</a>.