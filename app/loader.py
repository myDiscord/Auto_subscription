from os import PathLike
from pathlib import Path
import logging

import dill
from sklearn.base import BaseEstimator

# Logger creating
logger = logging.getLogger(__name__)

# Model name template
model_name_pattern = 'model_*.pkl'


def load_model(folder: PathLike) -> BaseEstimator:
    """Loads the latest model from the given folder."""

    # Get the list of models in the folder
    folder = Path(folder)
    model_files = list(folder.glob(model_name_pattern))
    logger.debug(f'Load model from {folder}.')

    # If there are models, load the last one
    if model_files:
        last_model = sorted(model_files)[-1]
        logger.debug(f'Model {last_model} is being loaded.')
        with open(last_model, 'rb') as file:
            model = dill.load(file)
        return model

    # Else raise an error
    else:
        raise FileNotFoundError('There are no models in the specified folder.')