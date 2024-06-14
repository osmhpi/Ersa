from pathlib import Path
from jupyter_client import MultiKernelManager
from tqdm import tqdm
import sys
import pandas as pd
import click


def get_output(client, msg_id):
    from queue import Empty

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
    except Empty as e:
        return None


def get_result(client):
    msg_id = client.execute("print(result)")
    return get_output(client, msg_id)


def get_duration(client):
    msg_id = client.execute("print(duration)")
    return get_output(client, msg_id)


import os
from pathlib import Path


def check_pid(pid):
    """Check For the existence of a unix pid."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def run_kernels_serial(code, num_kernels):
    # try:
    mkm = MultiKernelManager()
    kernels = [mkm.get_kernel(mkm.start_kernel()) for _ in range(num_kernels)]
    clients = [kernel.client() for kernel in kernels]
    pids = [kernel.provisioner.process.pid for kernel in kernels]
    Path("pids.txt").write_text("\n".join([str(x) for x in pids]))
    # print(pids)

    results = []
    durations = []
    dead = []

    for cl_index, client in enumerate(tqdm(clients, file=sys.stdout)):
        tqdm.write(f"starting {pids[cl_index]}")
        try:
            client.execute(code, reply=True, timeout=45)
        except TimeoutError:
            result = "kernel died"
            duration = 0
            print(f"is alive {kernels[cl_index].is_alive()}")
        else:
            result = get_result(client)
            duration = get_duration(client)

        results.append(result)
        durations.append(duration)

        # dead_kernels = [index for index, kernel in enumerate(kernels) if not kernel.is_alive()]
        # dead_kernels = [index for index,kernel in enumerate(kernels) if kernel.provisioner.process.poll() is not None]
        dead_kernels = [index for index, pid in enumerate(pids) if not check_pid(pid)]
        dead.append(dead_kernels)
        tqdm.write(f"{result} {duration} {dead_kernels}")
    # except KeyboardInterrupt:
    #     print("Received Keyboard Interrupt")
    # finally:
    mkm.shutdown_all()

    return results, durations, dead


@click.command
@click.option("--kernels", "-k", "-n", required=True, type=int)
@click.argument("benchmark", type=click.File("r"))
@click.argument("output", type=click.File("w"))
def main(kernels, benchmark, output):
    code = benchmark.read()
    accuracys, durations, dead_kernels = run_kernels_serial(code, kernels)
    df = pd.DataFrame(
        {
            "accuracy": accuracys,
            "duration": durations,
            "dead_kernels": dead_kernels,
        }
    )
    df.to_csv(output)


if __name__ == "__main__":
    main()
