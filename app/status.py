from enum import Enum


class Status(Enum):
    """Different service statuses."""

    OK = 'Service is running.'
    NO_MODEL = 'Model not loaded yet, please try again later.'
    MODEL_ERROR = 'Unable to load model.'
    ERROR = 'Unexpected error.'