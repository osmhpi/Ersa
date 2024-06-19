"""Run multi kernel benchmarks."""
import os
import sys
import time
from pathlib import Path
from queue import Empty

import click
import nbformat
import pandas as pd
from jupyter_client import MultiKernelManager
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
    msg_id = client.execute("print(result)")
    return get_output(client, msg_id)


def get_duration(client):
    """Get the value of the 'duration' variable from the kernel."""
    msg_id = client.execute("print(duration)")
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
        durations = []
        dead = []
        external_duration = []

        for cl_index, client in enumerate(tqdm(clients, file=sys.stdout)):
            tqdm.write(f"starting {pids[cl_index]}")
            start = time.time()
            try:
                for cell in code:
                    client.execute(cell, reply=True, timeout=45)
                end = time.time()
            except TimeoutError:
                result = "kernel died"
                duration = 0
            else:
                result = get_result(client)
                duration = get_duration(client)

            external_duration.append(end - start)
            results.append(result)
            durations.append(duration)

            dead_kernels = [
                index for index, pid in enumerate(pids) if not check_pid(pid)
            ]
            dead.append(dead_kernels)
            tqdm.write(f"{result} {duration} {dead_kernels}")
    except KeyboardInterrupt:
        print("Received Keyboard Interrupt")
    finally:
        mkm.shutdown_all()

    return results, durations, dead, external_duration


@click.command
@click.option("--kernels", "-k", "-n", required=True, type=int)
@click.argument(
    "benchmark", type=click.Path(exists=True, path_type=Path, dir_okay=False)
)
@click.argument("output", type=click.File("w"))
def main(kernels, benchmark: Path, output):
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
    accuracys, durations, dead_kernels, external_duration = run_kernels_serial(
        code, kernels
    )
    pd.DataFrame(
        {
            "accuracy": accuracys,
            "duration": durations,
            "dead_kernels": dead_kernels,
            "external_duration": external_duration,
        }
    ).to_csv(output)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
