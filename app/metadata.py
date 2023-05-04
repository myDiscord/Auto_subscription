from pydantic import BaseModel


class Metrics(BaseModel):
    """The structure of the model metrics."""

    roc_auc: float
    roc_auc_by_class: float
    accuracy: float
    precision: float
    recall: float
    f1: float


class Metadata(BaseModel):
    """Metadata structure."""

    name: str
    description: str
    version: float
    author: str
    model_type: str
    training_datetime: str
    threshold: float
    metrics: Metrics