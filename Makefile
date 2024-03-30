.DEFAULT_GOAL := default

install:
	@pip install -e .


run_api:
	uvicorn market.api.fast_price:app --reload

run_stlt:
	streamlit run app/app_ri.py
