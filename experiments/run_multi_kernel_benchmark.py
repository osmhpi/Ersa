"""Run multi kernel benchmarks."""

import copy
import csv
import datetime
import os
import subprocess
import sys
import time
from multiprocessing import Event, Process
from pathlib import Path
from queue import Empty

import click
import nbformat
import pandas as pd
import psutil
from jupyter_client import MultiKernelManager
from pynvml3.enums import TemperatureSensors
from pynvml3.pynvml import NVMLLib
from tqdm import tqdm


def get_output(client, msg_id):
    """Execute code in a kernel and get the result from stdout."""
    try:
        while True:
            message = client.get_iopub_msg(timeout=10)
            if message["parent_header"]["msg_id"] != msg_id:
                continue
            if message["header"]["msg_type"] != "stream":
                continue
            content = message["content"]
            if content["name"] != "stdout":
                continue
            return content["text"].strip()
    except Empty:
        return None


def get_result(client):
    """Get the value of the 'result' variable from the kernel."""
    return get_var(client, "result")


def get_start(client):
    """Get the value of the 'start' variable from the kernel."""
    return get_var(client, "start")


def get_end(client):
    """Get the value of the 'end' variable from the kernel."""
    return get_var(client, "end")


def get_var(client, name):
    """Get the value of the variable from the kernel."""
    msg_id = client.execute(f"print({name})")
    return get_output(client, msg_id)


def check_pid(pid):
    """Check For the existence of a unix pid."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def run_kernels_serial(code, num_kernels):
    """Run the given code serialy in each kernel."""
    try:
        mkm = MultiKernelManager()
        kernels = [mkm.get_kernel(mkm.start_kernel()) for _ in range(num_kernels)]
        clients = [kernel.client() for kernel in kernels]
        pids = [kernel.provisioner.process.pid for kernel in kernels]
        Path("pids.txt").write_text("\n".join([str(x) for x in pids]), encoding="utf-8")

        results = []
        starts = []
        ends = []
        dead = []
        external_starts = []
        external_ends = []

        for cl_index, client in enumerate(tqdm(clients, file=sys.stdout)):
            tqdm.write(f"starting {pids[cl_index]}")
            try:
                external_start = datetime.datetime.now()
                for cell in code:
                    client.execute(cell, reply=True, timeout=45)
                external_end = datetime.datetime.now()
            except TimeoutError:
                result = "kernel died"
                start = None
                end = None
                external_end = datetime.datetime.now()
            else:
                result = get_result(client)
                start = get_start(client)
                end = get_end(client)

            external_starts.append(external_start.isoformat())
            external_ends.append(external_end.isoformat())
            results.append(result)
            starts.append(start)
            ends.append(end)

            dead_kernels = [
                index for index, pid in enumerate(pids) if not check_pid(pid)
            ]
            dead.append(dead_kernels)
            tqdm.write(
                f"{result} {(external_end - external_start).total_seconds()} {dead_kernels}"
            )
    except KeyboardInterrupt:
        print("Received Keyboard Interrupt")
    finally:
        mkm.shutdown_all()

    return results, starts, ends, dead, external_starts, external_ends


def record_metrics(metrics_path, done):
    """Record metrics from GPU and CPU."""
    DELAY = 0.5
    with NVMLLib() as lib:
        print("Driver Version:", lib.system.get_driver_version())
        data = []
        device = lib.device[0]
        
        while not done.is_set():
            start = time.time()

            util = device.get_utilization_rates()
            gpu_memory = device.get_memory_info(version=2)
            host_memory = psutil.virtual_memory()
            swap_memory = psutil.swap_memory()
            data.append(
                [
                    datetime.datetime.now().isoformat(),
                    util.memory,
                    util.gpu,
                    device.get_power_usage(),
                    device.get_total_energy_consumption(),
                    device.get_temperature(TemperatureSensors.TEMPERATURE_GPU),
                    gpu_memory.total,
                    gpu_memory.reserved,
                    gpu_memory.free,
                    gpu_memory.used,
                    psutil.cpu_percent(),
                    host_memory.total,
                    host_memory.used,
                    host_memory.free,
                    host_memory.percent,
                    swap_memory.total,
                    swap_memory.used,
                    swap_memory.free,
                    swap_memory.percent,
                    swap_memory.sin,
                    swap_memory.sout,
                ]
            )

            end = time.time()
            time.sleep(max(DELAY - (end - start),0))

    pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "gpu_memory_util",
            "gpu_util",
            "power",
            "total_energy",
            "temperature",
            "gpu_memory_total",
            "gpu_memory_reserved",
            "gpu_memory_free",
            "gpu_memory_used",
            "cpu_util",
            "host_memory_total",
            "host_memory_used",
            "host_memory_free",
            "host_memory_percentage",
            "swap_memory_total",
            "swap_memory_used",
            "swap_memory_free",
            "swap_memory_percentage",
            "swap_memory_in",
            "swap_memory_out",
        ],
    ).to_csv(metrics_path)


@click.command
@click.option("--kernels", "-k", "-n", required=True, type=int)
@click.argument(
    "benchmark", type=click.Path(exists=True, path_type=Path, dir_okay=False)
)
@click.argument(
    "output", type=click.Path(path_type=Path, dir_okay=False, writable=True)
)
# @click.option('--framework',
#               type=click.Choice(['baseline', "nvshare","ersa"], case_sensitive=False))
def main(kernels, benchmark: Path, output: Path):
    """The main function."""
    if benchmark.suffix == ".py":
        code = [benchmark.read_text()]

    elif benchmark.suffix == ".ipynb":
        notebook = nbformat.read(str(benchmark), as_version=4)
        code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]

    else:
        raise ValueError(
            "Benchmark needs to be a python script (.py) or jupyter notebook (.ipynb)."
        )
    
    # env = copy.deepcopy(os.environ)

    # if framework != "ersa":
    #     env |= {"ERSA_DISABLE": "1"}
    # else:

    
    # if framework == "nvshare":
    #     env |= {"LD_PRELOAD": "libnvshare.so"}
    #     subprocess.run(
    #             env=env
    #         )

    done = Event()
    process = Process(
        target=record_metrics, args=(output.with_suffix(".metrics.csv"), done)
    )
    process.start()
    accuracys, starts, ends, dead_kernels, external_starts, external_ends = (
        run_kernels_serial(code, kernels)
    )
    done.set()
    process.join()
    pd.DataFrame(
        {
            "accuracy": accuracys,
            "start": starts,
            "end": ends,
            "dead_kernels": dead_kernels,
            "external_start": external_starts,
            "external_end": external_ends,
        }
    ).to_csv(str(output))


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
