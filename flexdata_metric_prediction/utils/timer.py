import logging
import time


class Timer:
    """Timer utility. Only for coarse-grained timing."""

    def __init__(self) -> None:
        self._start = None
        self.name = ""
        self.lap_counter = 0

    def start(self, name: str = ""):
        self.name = name
        self._start = time.perf_counter()

    def stop(self):
        if self._start is None:
            logging.warning(f"Timer was not started (name: {self.name})")
            return
        elapsed = time.perf_counter() - self._start
        self._report(elapsed)
        self.name = ""
        self.lap_counter = 0

    def lap(self):
        if self._start is None:
            logging.warning(f"Timer was not started (name: {self.name})")
            return
        elapsed = time.perf_counter() - self._start
        self._report(elapsed)
        self.lap_counter += 1
        self._start = time.perf_counter()

    def _report(self, elapsed):
        logging.info(f"[{self.name}] ({self.lap_counter}) Elapsed time: {elapsed:.2f} seconds.")
