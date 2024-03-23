.DEFAULT_GOAL := default

install:
	@pip install -e .


run_api
	uvicorn energy_price_pred.api.fast_price:app --reload
