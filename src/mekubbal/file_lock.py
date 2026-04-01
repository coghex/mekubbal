from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


@contextmanager
def locked_path(path: Path) -> Generator[Path, None, None]:
    """Acquire an exclusive file lock for read-modify-write on *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield path
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
