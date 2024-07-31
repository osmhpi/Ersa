import datetime
import logging
from pathlib import Path
import signal
import time

class StatsTracer(object):
    def __init__(self, ipython, log_path: Path = Path("stats-log.log")):
        self.ipython = ipython

        self.log_path = Path(log_path)
        self.log_file = self.log_path.open("a", encoding="UTF-8")
        if self.log_path.stat().st_size == 0:
            self.log_file.write("timestamp,event,data,cell_id\n")
            self.log_file.flush()
        
        self._register("pre_execute", self.pre_execute)
        self._register("post_execute", self.post_execute)
        self._register("pre_run_cell", self.pre_run_cell)
        self._register("post_run_cell", self.post_run_cell)

        self.log("extension_loaded")

    def log(self, event: str, data: str = "", cell_id: str=""):
        timestamp = datetime.datetime.now().isoformat()
        self.log_file.write(f"{timestamp},{event},{data},{cell_id}\n")
        self.log_file.flush()

    def __del__(self):
        self.log("__del__")
        self.log_file.close()

    def _register(self, event, callback):
        self.ipython.events.register(event, callback)

    def pre_execute(self):
        """pre_execute is like pre_run_cell,
        but is triggered prior to any execution.
        Sometimes code can be executed by libraries,
        etc. which skipping the history/display mechanisms,
        in which cases pre_run_cell will not fire.
        """
        self.log("pre_execute")

    def post_execute(self):
        """The same as pre_execute, post_execute is like post_run_cell,
        but fires for all executions, not just interactive ones.
        """
        self.log("post_execute")

    def pre_run_cell(self, info):
        gpu_tag = False
        try:
            gpu_tag = "tag: gpu" in info.raw_cell.split("\n")[0].strip()
        except Exception:
            pass
        self.log("pre_run_cell", "gpu" if gpu_tag else "", cell_id=info.cell_id)

    def post_run_cell(self, result):
        self.log("post_run_cell")


def load_ipython_extension(ipython):
    tracer=StatsTracer(ipython)
    print("execution tracer loaded")

    def goodbye(name, adjective):
        tracer.log("goodbye")

    import atexit

    atexit.register(goodbye, 'Donny', 'nice')

