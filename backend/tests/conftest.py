import os

os.environ.setdefault("TESTING", "true")
os.environ.setdefault("DATA_DIR", "/tmp/healthcare_rag_tests")
os.environ.setdefault("HF_HOME", "/tmp/healthcare_rag_tests/models")
os.environ.setdefault("PLANNER_MODE", "off")
