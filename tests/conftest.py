import os
import shutil
import sys
from pathlib import Path

# Make the project root importable when running `pytest` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_crewai_test_data: Path | None = None


def pytest_sessionstart(session):
    """Keep CrewAI's SQLite test data inside this run's isolated project folder."""
    global _crewai_test_data
    import appdirs

    _crewai_test_data = (
        Path(session.config.rootpath)
        / ".pytest-crewai-data"
        / str(os.getpid())
    )

    def _test_user_data_dir(appname=None, appauthor=None, version=None, roaming=False):
        path = _crewai_test_data
        if appauthor is not False and appauthor:
            path /= appauthor
        if appname:
            path /= appname
        if version:
            path /= version
        return str(path)

    appdirs.user_data_dir = _test_user_data_dir


def pytest_sessionfinish(session, exitstatus):
    """Remove only the generated CrewAI test database for this pytest process."""
    if _crewai_test_data is not None:
        shutil.rmtree(_crewai_test_data, ignore_errors=True)
