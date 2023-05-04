"""
Script to create a model.
"""

from typing import Tuple, Dict, Union
from pathlib import Path
from datetime import datetime
from os import PathLike

import pandas as pd
import dill
from lightgbm import LGBMClassifier
# for preparing and evaluating the model
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from feature_engine.selection import (
    DropFeatures, DropDuplicateFeatures,
    DropCorrelatedFeatures, DropConstantFeatures)
from feature_engine.encoding import RareLabelEncoder, OneHotEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.outliers import Winsorizer
from feature_engine.transformation import YeoJohnsonTransformer

import additional_data as add

# Model Information
RANDOM_SEED = 0
VERSION = 1.1
AUTHOR = 'Mykhailo Kafka'
NAME = 'SberAutopodpiska: target event prediction'
DESCRIPTION = ('Model for predicting whether the user will do one of the'
               'target actions "Order a call" or "Leave a request" to  site of the '
               'SberAutopodpiska service.')

# Specify the path to the data and to the folder with the models
DATA_FOLDER = Path('data')
SESSIONS_FILENAME = 'ga_sessions.csv'
HITS_FILENAME = 'ga_hits.csv'
MODELS_FOLDER = Path('models')

# Type for metadata
_metadata_type = Dict[str, Union[str, float, Dict[str, float]]]


def create_model(
    data_folder: Union[PathLike, None] = None,
    models_folder: Union[PathLike, None] = None
) -> None:
    """Creates, trains and saves the model."""

    # Getting data and creating a model
    X, y = _load_data(data_folder or DATA_FOLDER)
    model = _get_model()

    # Model metadata and metrics
    metadata = _get_metadata(model, X, y)

    # Training the model on all data
    model.fit(X, y)

    # Saving the model with metadata
    _save_model(model, metadata, models_folder or MODELS_FOLDER)


def _load_data(folder: PathLike) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads data from files with sessions and events."""

    sessions_file = Path(folder) / SESSIONS_FILENAME
    hits_file = Path(folder) / HITS_FILENAME

    # Upload files (if they exist)
    for file in (sessions_file, hits_file):
        if not file.exists():
            raise FileNotFoundError(f'Не найден файл {file}')
    sessions = pd.read_csv(sessions_file)
    hits = pd.read_csv(hits_file, usecols=['session_id', 'event_action'])

    # Get the target variation
    hits['target'] = hits['event_action'].isin(add.target_events)
    is_target_event = hits.groupby('session_id')['target'].any().astype(float)
    target = pd.Series(is_target_event, index=sessions['session_id'])

    return sessions, target.fillna(0.0)


def _get_model() -> BaseEstimator:
    """Returns the model with the hyperparameters found in the notebook
         with model research `notebooks/model_research.ipynb`.
         """

    return Pipeline(steps=[
        # Create additional features and
        # Bringing the dataframe to a convenient form
        ('indexer', FunctionTransformer(_set_index)),
        ('imputer', FunctionTransformer(_fill_missings)),
        ('engineer', FunctionTransformer(_create_features)),
        ('dropper', DropFeatures(['client_id', 'visit_date', 'visit_time',
                                  'device_screen_resolution'])),
        # Transformations of numerical variables
        ('normalization', YeoJohnsonTransformer()),
        ('outlier_remover', Winsorizer()),
        ('scaler', SklearnTransformerWrapper(StandardScaler())),
        # Transformations of categorical features
        ('rare_encoder', RareLabelEncoder(tol=0.03541, replace_with='rare')),
        ('onehot_encoder', OneHotEncoder(drop_last_binary=True)),
        ('bool_converter', FunctionTransformer(_converse_types)),
        # Removing duplicates and correlated features
        ('constant_dropper', DropConstantFeatures(tol=0.9767)),
        ('duplicated_dropper', DropDuplicateFeatures()),
        ('correlated_dropper', DropCorrelatedFeatures(threshold=0.8633)),
        # Best model with optimized hyperparameters
        ('model', LGBMClassifier(
            random_state=RANDOM_SEED, learning_rate=0.04197,
            boosting_type='gbdt', n_estimators=4700, reg_lambda=2.237,
            reg_alpha=8.335, num_leaves=16))])


def _set_index(data: pd.DataFrame, column: str = 'session_id') -> pd.DataFrame:
    """Sets the dataframe index to `column`."""

    data = data.copy()

    if column in data.columns:
        data = data.set_index(column)

    return data


def _fill_missings(data: pd.DataFrame) -> pd.DataFrame:
    """Fills in missing values:
    * the most common value for `device_screen_resolution`;
    * value '(nan)' in all other cases.
    """

    data = data.copy()

    if 'device_screen_resolution' in data.columns:
        # '414x896' - mod 'device_screen_resolution' as analyzed
        data['device_screen_resolution'] \
            .replace(add.missing_values, '414x896', inplace=True)

    return data.fillna('(nan)')


def _create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Creates new features from existing ones."""

    data = data.copy()

    # visit_date signs
    if 'visit_date' in data.columns:
        data['visit_date'] = data['visit_date'].astype('datetime64[ns]')
        data['visit_date_added_holiday'] = \
            data['visit_date'].isin(add.russian_holidays)
        # make numerical features strictly positive
        # for better handling in step with YeoJohnsonTransformer
        data['visit_date_weekday'] = data['visit_date'].dt.weekday + 1
        data['visit_date_weekend'] = data['visit_date'].dt.weekday > 4
        data['visit_date_day'] = data['visit_date'].dt.day + 1

    # visit_time features
    if 'visit_time' in data.columns:
        data['visit_time'] = data['visit_time'].astype('datetime64[ns]')
        data['visit_time_hour'] = data['visit_time'].dt.hour + 1
        data['visit_time_minute'] = data['visit_time'].dt.minute + 1
        data['visit_time_night'] = data['visit_time'].dt.hour < 9

    # utm_* signs
    if 'utm_medium' in data.columns:
        data['utm_medium_added_is_organic'] = \
            data['utm_medium'].isin(add.organic_mediums)
    if 'utm_source' in data.columns:
        data['utm_source_added_is_social'] = \
            data['utm_source'].isin(add.social_media_sources)

    # device_screen features
    if 'device_screen_resolution' in data.columns:
        name='device_screen_resolution'
        data[[name + '_width', name + '_height']] = \
            data[name].str.split('x', expand=True).astype(float)
        data[name + '_area'] = data[name + '_width'] * data[name + '_height']
        data[name + '_ratio'] = data[name + '_width'] / data[name + '_height']
        data[name + '_ratio_greater_1'] = data[name + '_ratio'] > 1

    # geo_city features
    if 'geo_city' in data.columns:
        data['geo_city_added_is_moscow_region'] = \
            data['geo_city'].isin(add.moscow_region_cities)
        data['geo_city_added_is_big'] = data['geo_city'].isin(add.big_cities)
        data['geo_city_is_big_or_in_moscow_region'] = \
            data['geo_city_added_is_moscow_region'] \
            | data['geo_city_added_is_big']
        data['geo_city_added_distance_from_moscow'] = \
            data['geo_city'].apply(add.get_distance_from_moscow)
        data['geo_city_added_distance_from_moscow_category'] = \
            data['geo_city_added_distance_from_moscow'] \
                .apply(_distance_category)

    return data


def _distance_category(distance: float) -> str:
    """Returns the distance category to Moscow."""

    if distance == -1:
        return 'no distance'
    elif distance == 0:
        return 'moscow'
    elif distance < 100:
        return '< 100 km'
    elif distance < 500:
        return '100-500 km'
    elif distance < 1000:
        return '500-1000 km'
    elif distance < 3000:
        return '1000-3000 km'
    else:
        return '>= 3000 km'


def _converse_types(data: pd.DataFrame) -> pd.DataFrame:
    """Casts variable types to float. First of all
    needed to convert bool values.
    """

    return data.astype(float)


def _get_metadata(model, X, y) -> _metadata_type:
    """Evaluate the model and return the metadata."""

    # Select the test part and evaluate the model on it
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=200_000, stratify=y, random_state=RANDOM_SEED)

    # Train the model and get the best
    # threshold for translating probabilities into classes
    model.fit(X_train, y_train)
    probas = model.predict_proba(X_test)[:, 1]
    threshold = _find_best_threshold(y_test, probas)

    return {
        'name': NAME,
        'description': DESCRIPTION,
        'version': VERSION,
        'author': AUTHOR,
        'model_type': model['model'].__class__.__name__,
        'training_datetime': str(datetime.now()),
        'threshold': threshold,
        'metrics': _get_metrics(y_test, probas, threshold)}


def _find_best_threshold(
        y_true: pd.Series,
        y_proba: pd.Series,
        iterations: int = 250,
        learning_rate: float = 0.05
) -> float:
    """
    Finds the best translation threshold for `y_proba` probabilities in class 1.
    """

    # Get the metrics
    def get_metric(threshold: float) -> float:
        prediction = (y_proba > threshold).astype(int)
        return roc_auc_score(y_true, prediction)

    direction = -1
    shift = 0.25

    best_threshold = 0.5
    best_metric = get_metric(best_threshold)

    # At each iteration
    for i in range(iterations):

        # Change the threshold
        threshold = best_threshold + direction * shift
        shift *= (1 - learning_rate)
        metric = get_metric(threshold)

        # And check if the metric has improved
        if metric > best_metric:
            best_threshold = threshold
            best_metric = metric
        else:
            direction *= -1

    return best_threshold


def _get_metrics(y_true, y_proba, threshold) -> Dict[str, float]:
    """Returns the model metrics for the given
    probabilities and the threshold of their transfer to a class.
    """

    prediction = (y_proba > threshold).astype(float)

    return {
        'roc_auc': roc_auc_score(y_true, y_proba),
        'roc_auc_by_class': roc_auc_score(y_true, prediction),
        'accuracy': accuracy_score(y_true, prediction),
        'precision': precision_score(y_true, prediction),
        'recall': recall_score(y_true, prediction),
        'f1': f1_score(y_true, prediction)}


def _save_model(
        model: BaseEstimator,
        metadata: _metadata_type,
        folder: PathLike
) -> None:
    """Saves the model with metadata to the models folder."""

    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    filename = f'model_{datetime.now():%Y%m%d%H%M%S}.pkl'

    model.metadata = metadata
    with open(folder / filename, 'wb') as file:
        dill.dump(model, file)


if __name__ == '__main__':
    create_model()