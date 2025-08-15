from pathlib import Path
def repo_root() -> Path: return Path(__file__).resolve().parents[1]
def outputs_root() -> Path: return Path("outputs")
def data_root() -> Path: return Path("data")
