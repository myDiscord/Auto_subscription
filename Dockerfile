FROM python:3.9

# Create a working folder and install libraries
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy project files
COPY ./app /code/app
COPY ./create_model.py /code/create_model.py
COPY ./additional_data.py /code/additional_data.py

# Starting the server with the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]