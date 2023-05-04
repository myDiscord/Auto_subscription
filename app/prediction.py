from pydantic import BaseModel


class Prediction(BaseModel):
    """
    The structure of an individual model prediction. `prediction` can be
    both the class (0 or 1) and the probability of a positive class.
    """

    session_id: str
    prediction: float