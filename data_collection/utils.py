import contextlib
import os

def safe_mkdir(dirname):
    with contextlib.suppress(FileExistsError):
        os.mkdir(dirname)