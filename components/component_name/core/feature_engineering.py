import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Union
import pandas as pd


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class FeatureEngineering:
    config_name: Dict[str, Union[float, List[str]]]

    def data_aggregation_for_fit(
            self, raw_data: Optional[Dict[str, List[Union[str, float, int]]], pd.DataFrame]
    ) -> pd.DataFrame:
        raw_df = self.__data_converting(raw_data)
        self.__data_consistency_check(raw_df)

        cleaned_raw_df = self.__drop_empty_cols(raw_df)
        prepared_features_target = self.__create_features_target(cleaned_raw_df)

        return prepared_features_target

    @staticmethod
    def __data_converting(raw_data: Dict[str, List[Union[str, float, int]]]):
        """Dict to pandas DataFrame
        Args:
            raw_data (dict) in format:
                {
                    'date_time': ['01.01.2021 08:00', '01.01.2021 09:00', '01.01.2021 10:00', '01.01.2021 11:00']
                    'feature_1': [0, 1, 2, 3],
                    'feature_2': [1, 2, 3, 4],
                    'feature_3': [-1, -2, -3, -4],
                    'feature_4': [4, 5, 6, 7],
                }
        Return:
            raw_input_df (pd.DataFrame): raw data in pandas DataFrame format
        """
        if isinstance(raw_data, dict):
            raw_df = pd.DataFrame.from_dict(raw_data, orient='columns')
        elif isinstance(raw_data, pd.DataFrame):
            raw_df = raw_data
        else:
            raise TypeError('Feature Engineering should have DataFrame or dict as input raw_data')

        return raw_df

    def __data_consistency_check(self, raw_df):
        missing_columns = set(self.config_name['raw_data_columns']) - set(raw_df.columns)
        if not missing_columns:
            return True
        else:
            raise KeyError(f'Columns: {missing_columns} not found in raw data')

    @staticmethod
    def __drop_empty_cols(raw_df):
        cols_before_drop = raw_df.columns.to_list()
        cleaned_raw_df = raw_df.dropna(how='all', axis=1)

        dropped_cols = list(set(cols_before_drop) - set(cleaned_raw_df.columns))
        LOGGER.debug(f'Drop empty columns: {dropped_cols}')
        return cleaned_raw_df

    def __create_features_target(self, raw_df):
        filtered_raw_df = self.__drop_outliers(raw_df)
        prepared_features_target = self.__add_new_features(filtered_raw_df)

        return prepared_features_target

    @abstractmethod
    def __drop_outliers(self, raw_df):
        """
        Drop rows with outliers based on self.config_name['outliers_info'].
        :param raw_df (pd.DataFrame): raw data for filtering rows with outliers
        :return filtered_raw_df: data without rows with outliers
        """
        ...

    @abstractmethod
    def __add_new_features(self, filtered_raw_df): ...
