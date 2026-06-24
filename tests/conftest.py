import os


def pytest_configure() -> None:
    os.environ["MALAB_DISABLE_REMOTE_TRACING"] = "true"
