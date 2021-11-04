import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class DataAccessObject:

    @staticmethod
    def get_local_csv(path: Path, separator: str = ','):
        csv_df = pd.read_csv(path, sep=separator)
        return csv_df

    @staticmethod
    def get_local_excel(path: Path):
        excel_df = pd.read_excel(path)
        return excel_df
