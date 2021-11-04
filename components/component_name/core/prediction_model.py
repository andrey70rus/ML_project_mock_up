from abc import ABC, abstractmethod
import joblib
from pathlib import Path
import logging
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class BaseModel (ABC):

    def copy(self):
        return self.copy()

    @staticmethod
    def load(path: Path):
        prediction_model = joblib.load(path)

        return prediction_model

    def save(self, path: Path):
        joblib.dump(self, path)

    @abstractmethod
    def fit(self, prepared_features, target): ...

    @abstractmethod
    def predict(self, prepared_features): ...


class XGBRegressorExtended (XGBRegressor, BaseModel, ABC):
    """
        Extended class for XGBRegressor model
    """
    pass
