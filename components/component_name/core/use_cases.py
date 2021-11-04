from abc import abstractmethod
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score


from prediction_model import XGBRegressorExtended
from feature_engineering import FeatureEngineering
from ..data_transfer_object import DataAccessObject
from settings import raw_data_path, random_state, target_name


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
    dao: DataAccessObject

    def run_use_case(self, report_path: Path):
        raw_data = self.dao.get_local_csv(raw_data_path)
        prepared_data = self.feature_engineering.data_aggregation_for_fit(raw_data)

        trained_model, metrics = self.__train_model(prepared_data)

        if report_path:
            self.__make_model_report(report_path)

        return trained_model, metrics

    def __train_model(self, prepared_data: pd.DataFrame):

        # TODO add cross-validation
        X_train, X_test, y_train, y_test = train_test_split(
            prepared_data.drop(target_name, axis=1), prepared_data[target_name],
            test_size=0.3, random_state=random_state
        )

        trained_model = XGBRegressorExtended.fit(X_train, y_train)
        metrics = self.__evaluate_model(trained_model, X_test, y_test)

        return trained_model, metrics

    def __make_model_report(self, report_path: Path):
        ...

    @staticmethod
    def __evaluate_model(model: XGBRegressorExtended, X_test:pd.Series, y_test: pd.Series):
        y_predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_predictions)
        r2 = r2_score(y_test, y_predictions)
        mse = mean_squared_error(y_test, y_predictions)
        explained_variance = explained_variance_score(y_test, y_predictions)

        return {'R2': r2, 'MAE': mae, 'MSE': mse, 'explained_variance': explained_variance}
