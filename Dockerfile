FROM tensorflow/tensorflow:2.16.1
WORKDIR /prod
COPY requirements_prod.txt requirements.txt
RUN pip install -r requirements.txt
COPY market market
COPY energy_price_pred energy_price_pred
COPY setup.py setup.py
RUN pip install .
COPY Makefile Makefile
CMD uvicorn market.api.fast_price:app --host 0.0.0.0 --port $PORT
