"""
Script for a quick check of the application's health.
Run from terminal:
```
python test_app.py
```
"""


import time
import json
from pathlib import Path
from pprint import pprint
from typing import Union
from urllib.request import urlopen, Request


# Server address and path to examples
app_url = 'http://127.0.0.1:8000'
examples_file = Path('data', 'examples.json')


def send_request(endpoint: str, data: Union[bytes, None] = None) ->None:
    """Sends a request to the server and prints the response."""

    start_time = time.time()
    if data is not None:
        request = Request(app_url + endpoint, data=data, method='POST',
                          headers={"Content-Type": "application/json"})
    else:
        request = Request(app_url + endpoint)
    with urlopen(request) as response:
        result = json.loads(response.read())
    pprint(result)
    print(f'Request time {time.time() - start_time} seconds')
    print('-' * 80)


def test_app() -> None:
    """Sends all possible requests to the server and prints the response."""

    # Loading examples
    with open(examples_file, 'r') as file:
        examples = json.load(file)

    print('-' * 80)
    print('Application request tests.')
    print('-' * 80)

    print('Status request.')
    send_request('/status')

    print('Metadata request.')
    send_request('/version')

    print('Request class prediction for one object.')
    data = json.dumps(examples[0]).encode("utf-8")
    send_request('/predict', data)

    print('Query class probability prediction for one object.')
    data = json.dumps(examples[1]).encode("utf-8")
    send_request('/predict_proba', data)

    print('Query class prediction for multiple objects.')
    data = json.dumps(examples).encode("utf-8")
    send_request('/predict_many', data)

    print('Query class probability prediction for a set of objects.')
    data = json.dumps(examples).encode("utf-8")
    send_request('/predict_proba_many', data)


if __name__ == '__main__':

    try:
        test_app()
    except:
        print('Unexpected error. Check if the server with the application is up.')