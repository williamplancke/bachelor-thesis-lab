"""Readable run names and all-or-nothing publication of saved artifacts."""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import re
from tempfile import TemporaryDirectory


def run_name(config, timestamp=None):
    """Keep experiment settings visible while retaining a unique timestamp suffix."""
    def slug(value):
        return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-") or "unknown"

    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    experiment = slug(config.get("experiment", "intent"))
    tokenizer = slug(config.get("tokenizer_instance", "word"))
    architecture = slug(config.get("architecture", "mean-pooling"))
    return (
        f"{experiment}__{architecture}__{tokenizer}-v{config['vocabulary_size']}"
        f"__emb{config['embedding_dim']}__seed{config['seed']}__{timestamp}"
    )


@contextmanager
def staged_run(run_dir):
    """Publish a run only after every artifact has been written successfully.

    Staging stays on the destination filesystem so directory rename is atomic.
    Ordinary errors and KeyboardInterrupt clean up the staging directory; a hard
    process kill may leave a hidden .pending-* directory, never a completed run.
    """
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pending-", dir=run_dir.parent) as directory:
        staging_dir = Path(directory)
        yield staging_dir
        if run_dir.exists():
            raise FileExistsError(f"Run already exists: {run_dir}")
        staging_dir.rename(run_dir)
