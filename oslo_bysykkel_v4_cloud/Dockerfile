FROM python:3.9.1

COPY docker-requirements.txt .

RUN pip install -r docker-requirements.txt --trusted-host pypi.python.org --no-cache-dir

COPY flows /flows

ENTRYPOINT [ "python", "flows/oslo_bike_etl.py" ]