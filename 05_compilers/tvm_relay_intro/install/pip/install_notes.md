Pip install notes (fallback)

1) Create env
python -m venv .venv
source .venv/bin/activate

2) Install
pip install -r requirements.txt

If apache-tvm fails:
- Your Python version might not have a prebuilt wheel.
- Options:
  A) use the Docker path (preferred)
  B) build TVM from source (Week 9 Day 2.5 if needed)

Sanity check
python -c "import tvm; print(tvm.__version__)"
