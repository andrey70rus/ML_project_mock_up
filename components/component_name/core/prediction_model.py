from abc import ABC, abstractmethod
import joblib
from pathlib import Path
import logging
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('PredictionModel')


class BaseModel (ABC):

    def copy(self):
        return self.copy()

    @staticmethod
    def load(path: Path):
        prediction_model = joblib.load(path)

        return prediction_model

    @abstractmethod
    def fit(self, model_features, target_value): ...


    def save(self, path: Path):
        joblib.dump(path, self.model)


class XGBRegressorExtended (XGBRegressor, BaseModel):
    """
        Extended class for XGBRegressor model
    """
