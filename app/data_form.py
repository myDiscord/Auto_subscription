from typing import Optional

from pydantic import BaseModel


class DataForm(BaseModel):
    """The data structure for which the prediction is made."""

    session_id: str
    client_id: Optional[str]
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: Optional[str]
    utm_medium: Optional[str]
    utm_campaign: Optional[str]
    utm_adcontent: Optional[str]
    utm_keyword: Optional[str]
    device_category: Optional[str]
    device_os: Optional[str]
    device_brand: Optional[str]
    device_model: Optional[str]
    device_screen_resolution: Optional[str]
    device_browser: Optional[str]
    geo_country: Optional[str]
    geo_city: Optional[str]