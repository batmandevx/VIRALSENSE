# ViralSense: one command per phase.  `make all` runs everything after the data is in data/raw/kim/.
PY := .venv/bin/python
RUN := $(PY) -m viralsense.pipeline --stages

.PHONY: setup data features jury models recommend report app test all

setup:            ## environment (macOS: brew install libomp ollama; ollama pull qwen2.5vl:3b)
	uv venv --python 3.11 && uv pip install -r requirements.lock && uv pip install -e '.[dev]' --no-deps

data:             ## Phase 1a: sample, history features, account split, labels
	$(PY) -m viralsense.data.prefetch
	$(RUN) prepare

features:         ## Phase 1b: CLIP / MiniLM / concepts / stats + report
	$(RUN) features
	$(PY) -m viralsense.features.report

jury:             ## Phase 2: local persona jury (cached, resumable), then analytics
	$(RUN) jury
	$(PY) -m viralsense.jury.report

models:           ## Phase 3: tuning, model grid, calibration, regression, clusters, ablation, robustness, controls
	$(PY) -m viralsense.models.tune --trials 25
	$(RUN) classify calibrate regress cluster ablation temporal control leakage compare

recommend:        ## Phase 4: posting-time and caption-style bandits, LLM caption variants
	$(RUN) bandit caption_bandit captions

app:              ## Phase 5: dashboard on http://localhost:8501
	.venv/bin/streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest

all: data features jury models recommend test
