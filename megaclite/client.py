"""This module implements the client side jupyter extension of megaclite."""
import argparse
import datetime
from itertools import chain
import logging
import os
import shlex
import subprocess
import sys
from io import BytesIO
from multiprocessing.connection import Client
from pathlib import Path
from threading import Thread
from typing import Optional

import dill
import ipywidgets
import toml
import torch
from IPython.display import clear_output, display
from IPython.core.magic import (
    Magics,
    cell_magic,
    line_magic,
    magics_class,
    needs_local_scope,
)
from pip._internal.operations import freeze

from .messages import (
    AbortJob,
    ClientInfo,
    JobInfo,
    JobResult,
    JobState,
    ShellJob,
    StdOut,
    StdErr,
    TrainingJob,
)
from . import __version__ as VERSION


def collect_client_info() -> ClientInfo:
    """Return a client info object with data from the current environment."""
    return ClientInfo(
        python_version=sys.version.split(" ", maxsplit=1)[0],
        user_name=os.getlogin(),
        packages=list(freeze.freeze()),
    )


COMPUTE_CONFIGS = ["1", "2", "3", "4", "7"]


@magics_class
class RemoteTrainingMagics(Magics):
    """Implements the IPython magic extension."""

    def __init__(self, shell):
        super().__init__(shell)
        print(f"loading megaclite version {VERSION}")
        print(shell)
        self.host: str = "127.0.0.1"
        self.port: str = 6001
        self.key: str = None
        self.message_box = None
        self.socket = None
        self.address = None
        self.disabled = False

        if "MEGACLITE_DISABLE" in os.environ:
            self.disabled = os.environ["MEGACLITE_DISABLE"] == "1"
            print("megaclite is disabled")

        megaclite_rc_path = Path(".megacliterc")

        if megaclite_rc_path.exists():
            megaclite_rc = toml.load(megaclite_rc_path)
            self.host = megaclite_rc.get("host", self.host)
            self.port = megaclite_rc.get("port", self.port)
            self.socket = megaclite_rc.get("socket", self.socket)

        if self.socket is not None:
            self.address = self.socket
        else:
            self.address = (self.host, self.port)
        print(self.address)
        if not self.disabled:
            self.apply_torch_patches()

    def apply_torch_patches(self):
        # don't apply the patch again, if we already did so
        if "HAS_GPU" in globals():
            return

        original_tensor_to = torch.Tensor.to
        original_module_to = torch.nn.modules.module.Module.to
        torch.cuda.set_device = lambda device: None

        namespace = self.shell.user_ns
        tensor_map = {}
        module_map = {}
        namespace["tensor_map"] = tensor_map
        namespace["module_map"] = module_map

        def apply_pending_tensor_moves():
            # print("apply_pending_tensor_moves", len(tensor_map))
            for tensor, device in tensor_map.items():
                # print("original tensor to from apply_pending_tensor_moves", device)
                original_tensor_to(tensor, device=device)

        def apply_pending_module_moves():
            # print("apply_pending_module_moves", len(module_map))
            for module, device in module_map.items():
                # print("original module to from apply_pending_module_moves", device)
                original_module_to(module, device=device)

        def reverse_tensor_moves():
            # print("apply_pending_tensor_moves", len(tensor_map))
            for tensor, _ in tensor_map.items():
                original_tensor_to(tensor, device="cpu")

        def reverse_module_moves():
            # print("apply_pending_module_moves", len(module_map))
            for module, _ in module_map.items():
                original_module_to(module, device="cpu")

        namespace["apply_pending_tensor_moves"] = apply_pending_tensor_moves
        namespace["apply_pending_module_moves"] = apply_pending_module_moves

        namespace["reverse_tensor_moves"] = reverse_tensor_moves
        namespace["reverse_module_moves"] = reverse_module_moves

        HAS_GPU = False

        def patched_tensor_to(*args, **kwargs):
            if HAS_GPU or (len(args) <= 1 and "device" not in kwargs):
                # print("original tensor to")
                return original_tensor_to(*args, **kwargs)

            tensor = args[0]
            device = kwargs["device"] if "device" in kwargs else args[1]
            tensor_map[tensor] = device
            return tensor

        def patched_module_to(*args, **kwargs):
            if HAS_GPU or (len(args) <= 1 and "device" not in kwargs):
                # print("original module to")
                return self.original_module_to(*args, **kwargs)
            tensor = args[0]
            device = kwargs.get("device", args[1])
            module_map[tensor] = device
            return tensor

        namespace["torch"].Tensor.to = patched_tensor_to
        namespace["torch"].nn.modules.module.Module.to = patched_module_to

        namespace["torch"].cuda.is_available = lambda: True
        print("patch applied")

    def print(self, value: str):
        """Print a message to the currently active message box."""
        self.message_box.value = value

    def init_print(self):
        """Initialize a new output message box."""
        clear_output()
        self.message_box = ipywidgets.HTML()
        display(self.message_box)

    def send_job(self, job, on_success: Optional[callable] = None):
        """Send the job to the server and process responses."""
        conn = Client(self.address)
        try:
            logging.info("sending job start")
            conn.send(job)
            logging.info("sending job finished")
            job_uuid = None

            while message := conn.recv():
                if isinstance(message, JobInfo):
                    job_uuid = message.uuid
                    if message.state == JobState.PENDING:
                        self.print(
                            f"<i>You are number {message.no_in_queue + 1} in the queue.<i>"
                        )
                    elif message.state == JobState.STARTED:
                        logging.info("processing job started")
                        self.print("<i>Job started executing.<i>")
                    elif message.state == JobState.FINISHED:
                        logging.info("processing job finished")
                        self.print("<i>Retrieving weights from remote.<i>")
                        if on_success:
                            logging.info("retrieving weights from remote started")
                            result = conn.recv()
                            logging.info("retrieving weights from remote finished")
                            on_success(result)
                    elif message.state == JobState.FAILED:
                        self.print("<i>Job failed.<i>")
                        raise RuntimeError("Remote training job failed!")

                    if message.state.exited:
                        self.print(
                            f"<i>Job exited with status {str(message.state)}.<i>"
                        )
                        break
                elif isinstance(message, (StdOut, StdErr)):
                    print(message.line, end="")
        except KeyboardInterrupt:
            # pylint: disable=used-before-assignment
            self.print(f"<i>Aborting job with uuid {job_uuid}.<i>")
            conn.send(AbortJob(uuid=job_uuid))
            result = conn.recv()
            if result.state == JobState.ABORTED:
                self.print(f"<i>Job with uuid {job_uuid} was aborted.<i>")

    def send_training_job(self, cell: str, mig_slices: int):
        """Create, preprocess, send, and postprocess a training job."""

        file = BytesIO()
        logging.info("pickling start")

        dill.dump_module(file)

        self.print(f"<i>Sending {len(file.getbuffer())/2**20:.0f}MB of state.<i>")
        job = TrainingJob(
            cell,
            file.getvalue(),
            client=collect_client_info(),
            mig_slices=mig_slices,
        )

        logging.info("pickling finished")

        def success_handler(result: JobResult):
            logging.info("loading weights started")
            results_file = BytesIO(result.result)
            dill.load_module(results_file)

            logging.info("loading weights finished")

        self.send_job(job=job, on_success=success_handler)

    @line_magic
    def notebook(self, line):
        return self.shell

    @cell_magic
    def without_cuda(self, line, cell):
        torch.cuda.is_available = lambda: False
        # print(cell)
        result = self.shell.run_cell(cell)
        # exec(cell, self.shell.user_global_ns)
        torch.cuda.is_available = lambda: True
        return result

    @line_magic
    def sync_cwd(self, line):
        excludes = [".venv", "venv", ".git"]
        active_venv = os.environ.get("VIRTUAL_ENV")
        if active_venv:
            excludes.append(active_venv)

        excludes = [f'"{item}"' for item in excludes]
        excludes = list(chain(*zip(["--exclude"] * len(excludes), excludes)))
        server_wd_path = f"/tmp/megaclite/wd/{os.getlogin()}"
        server = "gx06"
        args = ["rsync", "-hazupEh", *excludes, ".", f"{server}:{server_wd_path}"]
        print(" ".join(args))
        process = subprocess.Popen(
            args,
            text=True,
        )
        out = ipywidgets.Output()
        display(out)

        def on_done():
            process.wait()
            out.append_stdout("done")

        thread = Thread(target=on_done)
        thread.start()

    @line_magic
    def run_remote(self, line):
        """Use: %remote_config <host> <port> <key>"""
        self.init_print()
        self.print(f"<i>executing command `{line}` on remote host</i>")
        job = ShellJob(command=line, client=collect_client_info())
        print(job.client.packages)
        self.send_job(job=job)

    @line_magic
    def tag_benchmark(self, line):
        logging.basicConfig(
            format=f"%(asctime)s,%(message)s,{VERSION},{line}",
            filename="log.log",
            encoding="utf-8",
            level=logging.INFO,
        )

    logging.Formatter.formatTime = (
        lambda self, record, datefmt=None: datetime.datetime.now().isoformat()
    )

    @cell_magic
    def train_remote(self, line, cell):
        """Use: %%remote <model> [<compute-slices>]"""
        self.init_print()

        if self.disabled:
            self.print("megaclite is disabled")
            return self.shell.run_cell(cell)
        parser = argparse.ArgumentParser(description="Remote training job args.")
        parser.add_argument("compute", choices=COMPUTE_CONFIGS, nargs="?")

        args = parser.parse_args(shlex.split(line))

        self.print(
            f"<i>training on a remote gpu</i>"
            + f"<br><i>MIG: requesting <b>{args.compute}</b> compute slices<i>"
        )
        self.send_training_job(
            cell,
            int(args.compute) if args.compute else None,
        )


def load_ipython_extension(ipython):
    """Register the megaclite magic with ipython."""
    magics = RemoteTrainingMagics(ipython)
    ipython.register_magics(magics)
