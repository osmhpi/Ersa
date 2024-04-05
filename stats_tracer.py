import datetime
import logging
from pathlib import Path

# class CsvFormatter(logging.Formatter):
#     def __init__(self):
#         super().__init__()
#         self.writer = csv.writer(self.output, quoting=csv.QUOTE_ALL)

#     def format(self, record):
#         self.writer.writerow([datetime.datetime.now().isoformat(), record.msg])
#         data = self.output.getvalue()
#         self.output.truncate(0)
#         self.output.seek(0)
#         return data.strip()
    
#     def formatTime(self, record, datefmt):
#         return 
class StatsTracer(object):
    def __init__(self, ipython):
        self.ipython = ipython
        # logging.basicConfig(level=self.log)
        # self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(self.log)
        log_path = Path("stats-log2.log")
        self.log_file = log_path.open("a",encoding="UTF-8")
        if log_path.stat().st_size == 0:
            self.log_file.write("timestamp,event,data\n")
            self.log_file.flush()
        self._register("pre_execute", self.pre_execute)
        self._register("post_execute", self.post_execute)
        self._register("pre_run_cell", self.pre_run_cell)
        self._register("post_run_cell", self.post_run_cell)

        # formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        # self.logger.basicConfig(
        #     format=f"%(asctime)s,%(message)s",
        #     filename="stats-log.log",
        #     encoding="utf-8",
        #     level=self.log,

        # )
        # self.logger.Formatter.formatTime = (
        #    lambda self, record, datefmt=None: datetime.datetime.now().isoformat()
        # )
        self.log("extension_loaded")

    def log(self, event: str, data: str=""):
        timestamp=datetime.datetime.now().isoformat()
        self.log_file.write(f"{timestamp},{event},{data}\n")
        self.log_file.flush()
    
    def __del__(self):
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
        self.log("pre_run_cell", "gpu" if gpu_tag else "")
 
    def post_run_cell(self, result):
        self.log("post_run_cell")


def load_ipython_extension(ipython):
    StatsTracer(ipython)
    print("execution tracer loaded")

