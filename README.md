# Prediction of targeted actions on the site 'SberAutosubscription'

The project as a final task for the course 'Introduction to Data Science' from Skillbox.

**Goal** - predict user's performance of one of the target actions - 'Leave a request' or 'Request a call' based on visit time 'visit_\*', advertising tags 'utm_\*', device characteristics 'device_\*' and location ' geo_\*'.

Target **metrics**: `roc-auc` > 0.65, prediction time no more than 3 seconds.


## Notebooks

Notebooks with exploratory data analysis and research of models for the task are placed in a separate folder `notebooks`. A file with additional data `additional_data.py` was also copied to them for independence from the rest of the project.

## Model training

To create a model, you need to run the script `create_model.py` (file `additional_data.py` and folders with data `data` and models `models` must be in the same directory).

Or, you can create a model by running a laptop with model studies.

## Run application

The model is designed as a separate application located in the `app` folder. To run it, you need to install the libraries from `requirements.txt` and run the command in the root folder of the project:

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

Also, the application can be run using Docker. To do this, you need to execute commands in the root folder of the project to build the image and run `docker-compose` (possibly with `sudo`):
```
docker-compose build
docker-compose-up

# to disable docker-compose
docker-compose down
```
The models folder `models` is mounted as an external volume in `docker-compose.yml`.

In either case, the application will be available at `http://127.0.0.1:8000`.

## API Methods

To work with the application, you can use queries:
+ `/status` (get) - to get the service status;
+ `/version` (get) - to get the version and metadata of the model;
+ `/predict` (post) - to predict the class of one object;
+ `/predict_many` (post) - to predict the class of a set of objects;
+ `/predict_proba` (post) - to predict the probability of a positive class of one object;
+ `/predict_proba_many` (post) - to predict the probability of a positive class of a set of objects.

All of these methods can be quickly tested using the `test_app.py` script.
