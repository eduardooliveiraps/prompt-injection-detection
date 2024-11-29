
install-llm-guard:
	pip install -e ./llm_guard

run-st:
	streamlit run streamlit-app/llm_guard_app.py

install-dev:
	pip install -r requirements.txt