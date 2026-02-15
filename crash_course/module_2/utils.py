import os
from pathlib import Path


def where_am_i(file: str) -> Path:
    """
    return a path to the dir that file is in. file should be __file__
    """
    return Path(file).parent


def find_a_pdf(dir: Path) -> Path:
    """
    returns the path to the first pdf I find in dir
    """
    for filename in os.listdir(dir):
        if filename.endswith(".pdf"):
            return dir / filename
    raise RuntimeError(f"error unable to find pdf in dir {dir}")
