# Dataset source

The project uses UCI Machine Learning Repository's Dry Bean dataset (dataset ID 602):
https://archive.ics.uci.edu/dataset/602/dry%2Bbean

`train.py` fetches it through the `ucimlrepo` client when run without `--data-csv`.
The experiment's held-out labeled test data is stored at the project root as `test_data.csv`.
The dataset is distributed under CC BY 4.0; retain attribution to UCI and DOI 10.24432/C50S4B.
