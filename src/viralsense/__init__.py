"""ViralSense: pre-publication Instagram engagement class prediction."""
import os

# On macOS torch and xgboost each bundle their own libomp; with both loaded, multi-threaded
# OpenMP regions segfault. One OpenMP thread avoids it (CLIP runs on MPS, RF uses joblib threads).
# Must run before torch or xgboost is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
