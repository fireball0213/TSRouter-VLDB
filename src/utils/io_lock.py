import os
import pickle
import tempfile
from contextlib import contextmanager

import fcntl


@contextmanager
def file_lock(lock_path: str):
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "a+") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


def atomic_pickle_dump(obj, target_path: str):
    target_dir = os.path.dirname(target_path) or "."
    os.makedirs(target_dir, exist_ok=True)
    lock_path = target_path + ".lock"
    with file_lock(lock_path):
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".pkl", dir=target_dir)
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(obj, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)