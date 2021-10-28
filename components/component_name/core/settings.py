from pathlib import Path

raw_data_path = Path('D://Documents//local_data.csv')
raw_data_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']

outliers_info = {
    'feature_1': (None, 10),
    'feature_2': (-5, None),
    'feature_3': (0, 5),
}
