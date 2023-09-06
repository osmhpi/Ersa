"""This module implements the server for the remote GPU extension."""
import hashlib
import os
import re
import subprocess
import tempfile
import uuid
from datetime import datetime
from multiprocessing import Process, Queue
from multiprocessing.connection import Connection, Listener
from pathlib import Path
from typing import Optional

import click
from pynvml3.device import MigDevice
from pynvml3.enums import ComputeInstanceProfile, GpuInstanceProfile
from pynvml3.pynvml import NVMLLib

from .messages import (
    AbortJob,
    JobInfo,
    JobResult,
    JobState,
    ShellJob,
    StdOut,
    TrainingJob,
)

EXCLUDED_PACKAGES = ["megaclite", ".*pynvml3"]
ADDITIONAL_PACKAGES = ["click"]


class MigSlice:
    """Context manager to aquire a mig slice (gpu+compute instance)."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        device_id: int,
        gi_profile: GpuInstanceProfile,
        ci_profile: ComputeInstanceProfile,
    ) -> None:
        self.device_id = device_id
        self.gi_profile = gi_profile
        self.ci_profile = ci_profile
        self.uuid = None
        self.device = None
        self.lib = None
        self.compute_instance = None
        self.gpu_instance = None
        self.mig_device = None

    def __enter__(self):
        self.lib = NVMLLib()
        self.lib.open()
        self.device = self.lib.device.from_index(self.device_id)
        self.device.mig_version = 1
        self.uuid = self._create_mig_slice()
        return self

    def __exit__(self, *argc, **kwargs):
        self.compute_instance.destroy()
        self.gpu_instance.destroy()
        self.lib.close()

    def _create_mig_slice(self):
        """Create the requested gpu and compute slice.

        Returns the uuid of the corresponding, created mig device.
        """
        print("requesting", self.gi_profile, self.ci_profile)
        print(
            "capacity", self.device.get_gpu_instance_remaining_capacity(self.gi_profile)
        )
        self.gpu_instance = self.device.create_gpu_instance(self.gi_profile)
        print(
            "remaining capacity after creating",
            self.device.get_gpu_instance_remaining_capacity(self.gi_profile),
        )
        self.compute_instance = self.gpu_instance.create_compute_instance(
            self.ci_profile
        )
        self.mig_device: MigDevice = self.gpu_instance.get_mig_device()
        mig_uuid = self.mig_device.get_uuid()
        print("mig uuid", mig_uuid)
        return mig_uuid


def install_python_version(version: str):
    """Install the requested python version."""
    # shell injection waiting to happen :)
    subprocess.run(["pyenv", "install", version, "-s"], check=True)


def get_tmp_dir(sub_dir=None):
    """Create a new temporary directory."""
    if sub_dir is None:
        sub_dir = datetime.now().isoformat()
    tmp_path = Path(tempfile.gettempdir(), "megaclite", sub_dir)
    tmp_path.mkdir(exist_ok=True, parents=True)
    return tmp_path


def get_venv(tmp_dir: Path) -> Path:
    """Return path to venv."""
    return tmp_dir / "venv"


def get_pip(tmp_dir: Path) -> Path:
    """Return path to pip."""
    return get_venv(tmp_dir) / "bin/pip"


def get_python(tmp_dir: Path) -> Path:
    """Return path to python interpreter."""
    return get_venv(tmp_dir) / "bin/python"


def get_state_file(tmp_dir: Path) -> Path:
    """Return path to state file."""
    return tmp_dir / "state.pkl"


def get_cell_file(tmp_dir: Path) -> Path:
    """Return path to cell file."""
    return tmp_dir / "cell.py"


def get_output_file(tmp_dir: Path) -> Path:
    """Return path to output."""
    return tmp_dir / "output.pt"


def get_python_with_version(version: str) -> Path:
    """Return a path to a python interpreter with the specified version."""
    return Path.home() / f".pyenv/versions/{version}/bin/python3"


def create_venv_with_requirements(version, requirements: list[str]):
    """Create a new venv with the requested python version and packages."""
    print("creating venv with python version", version)

    requirements = [
        r
        for r in requirements
        if re.search(f"({'|'.join(EXCLUDED_PACKAGES)})", r.split("==")[0]) is None
    ]
    requirements.extend(ADDITIONAL_PACKAGES)
    message = hashlib.sha256()
    message.update(version.encode())
    for req in sorted(requirements):
        message.update(req.encode())

    tmp_path = get_tmp_dir(message.hexdigest())
    if get_venv(tmp_path).exists():
        return tmp_path

    print("creating venv in", str(get_venv(tmp_path)))
    subprocess.run(
        [get_python_with_version(version), "-m", "venv", str(get_venv(tmp_path))],
        check=True,
    )
    print("installing user packages")
    subprocess.run(
        [str(get_pip(tmp_path)), "install", "-r", "/dev/stdin"],
        input="\n".join(requirements),
        text=True,
        check=True,
    )
    print("installing megaclite")
    subprocess.run(
        [str(get_pip(tmp_path)), "install", "."],
        text=True,
        check=True,
    )
    return tmp_path


# pylint: disable=too-few-public-methods
class RemoteStdout:
    """Wraps a multiprocessing.connection.Connection to be file compatible."""

    def __init__(self, conn: Connection) -> None:
        self.conn = conn

    def write(self, text):
        """Send the text back to the client."""
        self.conn.send(StdOut(text))


def execute_in_subprocess(tmp_dir: Path, job: TrainingJob, conn: Connection, gpu=None):
    """Setup the subprocess execution with stdout redirect."""

    state_file = get_state_file(tmp_dir)
    cell_file = get_cell_file(tmp_dir)
    output_file = get_output_file(tmp_dir)

    state_file.write_bytes(job.state)
    cell_file.write_text(job.cell)
    print(get_python(tmp_dir))
    if gpu is not None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        print("CUDA_VISIBLE_DEVICES", gpu)
    else:
        env = os.environ

    with subprocess.Popen(
        [
            get_python(tmp_dir),
            "-m",
            "megaclite._runtime",
            str(state_file),
            str(cell_file),
            job.model_name,
            str(output_file),
        ],
        stdout=subprocess.PIPE,
        text=True,
        cwd=str(tmp_dir),
        env=env,
    ) as process:
        conn.send(JobInfo(state=JobState.STARTED, no_in_queue=0, uuid=job.uuid))

        for line in iter(process.stdout.readline, ""):
            conn.send(StdOut(line))

    conn.send(JobInfo(state=JobState.FINISHED, no_in_queue=0, uuid=job.uuid))
    conn.send(JobResult(result=output_file.read_bytes()))


def execute_shell_script(tmp_dir: Path, job: ShellJob, conn: Connection):
    """Run the specified shell script in a subprocess and stream the output."""
    with subprocess.Popen(
        ["/bin/bash", "-c", job.command],
        stdout=subprocess.PIPE,
        text=True,
        cwd=str(tmp_dir),
    ) as process:
        conn.send(JobInfo(state=JobState.STARTED, no_in_queue=0, uuid=job.uuid))

        for line in iter(process.stdout.readline, ""):
            conn.send(StdOut(line))

    conn.send(JobInfo(state=JobState.FINISHED, no_in_queue=0, uuid=job.uuid))


def worker_main(queue, gpus):
    """The main worker thread."""
    while True:
        message, conn = queue.get()
        install_python_version(message.client.python_version)
        tmp_dir = create_venv_with_requirements(
            message.client.python_version, message.client.packages
        )
        if isinstance(message, TrainingJob):
            if message.mig_slices is not None:
                with MigSlice(
                    device_id=1,
                    gi_profile=GpuInstanceProfile.from_int(message.mig_slices),
                    ci_profile=ComputeInstanceProfile.from_int(message.mig_slices),
                ) as mig_slice:
                    execute_in_subprocess(tmp_dir, message, conn, mig_slice.uuid)
            else:
                gpu = gpus.get()
                execute_in_subprocess(tmp_dir, message, conn, gpu)
                gpus.put(gpu)
        elif isinstance(message, ShellJob):
            execute_shell_script(tmp_dir, message, conn)


# pylint: disable=too-many-locals
@click.command()
@click.option("-h", "--host", default="127.0.0.1", type=str)
@click.option("-p", "--port", default=6001, type=int)
@click.option("-w", "--workers", default=1, type=int)
@click.option("-s", "--socket", default=None, type=str)
@click.option("-g", "--gpu", multiple=True, default=["0"])
def main(host: str, port: int, workers: int, socket: Optional[str], gpu: list[str]):
    """The main function"""
    if socket is not None:
        listener = Listener(socket)
    else:
        listener = Listener((host, port))

    jobs = Queue()
    worker_processes = []
    gpus = Queue()
    for gpu_item in gpu:
        gpus.put(gpu_item)

    for _ in range(workers):
        new_worker = Process(
            target=worker_main,
            args=(
                jobs,
                gpus,
            ),
        )
        new_worker.start()
        worker_processes.append(new_worker)

    # if we don't have 1 GPU per worker, we will overbook gpus
    for index in range(max(len(gpu) - workers, 0)):
        gpus.put(gpu[index % len(gpu)])

    while True:
        try:
            conn = listener.accept()
            message = conn.recv()
            if isinstance(message, TrainingJob):
                print(
                    "got new TrainingJob",
                    listener.last_accepted,
                    f"#{jobs.qsize()} in queue",
                )
                job_uuid = str(uuid.uuid4())
                message.uuid = job_uuid
                conn.send(
                    JobInfo(
                        state=JobState.PENDING, no_in_queue=jobs.qsize(), uuid=job_uuid
                    )
                )
            elif isinstance(message, ShellJob):
                print(
                    "got new BashJob",
                    listener.last_accepted,
                    f"#{jobs.qsize()} in queue",
                )
                job_uuid = str(uuid.uuid4())
                message.uuid = job_uuid
                conn.send(
                    JobInfo(
                        state=JobState.PENDING, no_in_queue=jobs.qsize(), uuid=job_uuid
                    )
                )
            elif isinstance(message, AbortJob):
                print("aborting job with uuid", message.uuid)
            jobs.put((message, conn))
        except KeyboardInterrupt:
            print("got Ctrl+C, cleaning up")
            listener.close()
            for worker in worker_processes:
                worker.join()
            break
        except Exception:  # pylint: disable=broad-exception-caught
            listener.close()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
