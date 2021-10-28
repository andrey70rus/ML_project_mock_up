from abc import abstractmethod
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from prediction_model import XGBRegressorExtended
from feature_engineering import FeatureEngineering
from ..data_transfer_object import DataTransferObject
from settings import raw_data_path


class UseCase:
    @abstractmethod
    def run_use_case(self, report_path: Path): ...


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class FitPredictionModel(UseCase):
    """
    UseCase for fitting prediction model based on ML algorithm
    """
    prediction_model: XGBRegressorExtended
    feature_engineering: FeatureEngineering
    dto: DataTransferObject

    def run_use_case(self, report_path: Path):
        raw_data = self.dto.get_local_csv(raw_data_path)
        prepared_data = self.feature_engineering.data_aggregation_for_fit(raw_data)

        trained_model = self.__train_model(prepared_data)

        if report_path:
            self.__make_model_report(report_path)

        return trained_model

    def __train_model(self, prepared_data: pd.DataFrame):
        ...

    def __make_model_report(self, report_path: Path):
        ...

    def __evaluate_model(self):
        ...
