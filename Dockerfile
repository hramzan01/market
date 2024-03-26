FROM python:3.11.8-bullseye
WORKDIR /prod
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY market market
COPY energy_price_pred energy_price_pred
COPY setup.py setup.py
RUN pip install .
COPY Makefile Makefile
COPY raw_data raw_data
CMD uvicorn market.api.fast_price:app --host 0.0.0.0 --port $PORT
