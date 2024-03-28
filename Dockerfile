FROM python:3.11.8-bullseye
WORKDIR /prod
COPY requirements_prod.txt requirements.txt
RUN pip install -r requirements.txt
COPY market/market market

COPY setup.py setup.py
RUN pip install .
COPY Makefile Makefile

CMD uvicorn market.api.fast_price:app --host 0.0.0.0 --port $PORT
