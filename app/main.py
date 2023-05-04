import logging
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder

from .status import Status
from .loader import load_model
from .metadata import Metadata
from .prediction import Prediction
from .data_form import DataForm


# Creating a logger
log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logger = logging.getLogger(__name__)

# Create an application
app = FastAPI()
status = Status.NO_MODEL
model = None
logger.debug(f'Application created. Status: {status.name}.')


@app.get('/status')
def get_status() -> str:
    """Returns the status of the service."""

    logger.debug(f'Service status request. Status: {status.name}.')
    return status.value


def _check_status() -> None:
    """Checks the status of the service. If it's not running it raises an error."""

    if status != Status.OK:
        logger.error(
            f'The request could not be completed, the service is not ready to run.')
        raise HTTPException(status_code=404, detail=status.value)


@app.get('/version', response_model=Metadata)
def get_metadata() -> Metadata:
    """Returns model metadata."""

    logger.debug(f'Service status request. Status: {status.name}.')
    _check_status()
    try:
        metadata = model.metadata
    except AttributeError:
        logger.error(f'Metadata request denied: model has no metadata.')
        raise HTTPException(status_code=404, detail='Model has no metadata.')
    else:
        return Metadata.parse_obj(metadata)


def _predict(data: pd.DataFrame, return_proba: bool = False) -> pd.Series:
    """Predicts the class or probability of a positive class."""

    # Got the threshold
    try:
        threshold = model.metadata['threshold']
    except:
        threshold = 0.5

    # Making a prediction
    try:
        prediction = model.predict_proba(data)[:, 1]
    except AttributeError:
        prediction = model.predict(data)
    else:
        if not return_proba:
            prediction = (prediction > threshold).astype(float)

    return prediction


@app.post('/predict', response_model=Prediction)
def predict_class(data_form: DataForm) -> Prediction:
    """Returns the predicted class for a single object."""

    logger.debug(f'Service status request. Status: {status.name}.')
    _check_status()

    # Get prediction
    data = pd.DataFrame(jsonable_encoder([data_form]))
    prediction = _predict(data, return_proba=False)[0]
    logger.info(f'{prediction} - prediction for '
                f'`sessions_id`={data_form.session_id}.')

    return Prediction(session_id=data_form.session_id, prediction=prediction)


@app.post('/predict_proba', response_model=Prediction)
def predict_proba(data_form: DataForm) -> Prediction:
    """Returns the probability of a positive class for a single object."""

    logger.debug(f'Request to predict the probability of a positive class. '
                 f'Status: {status.name}.')
    _check_status()

    # Get prediction
    data = pd.DataFrame(jsonable_encoder([data_form]))
    prediction = _predict(data, return_proba=True)[0]
    logger.info(f'{prediction} - prediction for '
                f'`sessions_id`={data_form.session_id}.')

    return Prediction(session_id=data_form.session_id, prediction=prediction)


@app.post('/predict_many', response_model=List[Prediction])
def predict_classes(data_forms: List[DataForm]) -> List[Prediction]:
    """Returns the probability of a positive class for a set of objects."""

    logger.debug(f'Service status request. Status: {status.name}.')
    _check_status()

    # Get prediction
    data = pd.DataFrame(jsonable_encoder(data_forms))
    predictions = _predict(data, return_proba=False)
    logger.info(f'{predictions} - predictions for a group of objects.')

    results = list()
    for data_form, prediction in zip(data_forms, predictions):
        result = Prediction(
            session_id=data_form.session_id, prediction=prediction)
        results.append(result)

    return results


@app.post('/predict_proba_many', response_model=List[Prediction])
def predict_probas(data_forms: List[DataForm]) -> List[Prediction]:
    """Returns the probability of a positive class for a set of objects."""

    logger.debug(f'Service status request. Status: {status.name}.')
    _check_status()

    # Get prediction
    data = pd.DataFrame(jsonable_encoder(data_forms))
    predictions = _predict(data, return_proba=True)
    logger.info(f'{predictions} - predictions for a group of objects.')

    results = list()
    for data_form, prediction in zip(data_forms, predictions):
        result = Prediction(
            session_id=data_form.session_id, prediction=prediction)
        results.append(result)

    return results


# Model loading
try:
    model = load_model('models')
except FileNotFoundError:
    status = Status.MODEL_ERROR
    logger.error(
        f'Unable to load the model. Status: {status.name}.', exc_info=True)
except Exception:
    status = Status.ERROR
    logger.error(
        f'Unable to load the model. Status: {status.name}.', exc_info=True)
else:
    status = Status.OK
    logger.debug(f'Model loaded. Status: {status.name}.')