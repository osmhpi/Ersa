import time
from pynvml3.pynvml import NVMLLib
from pynvml3.enums import SamplingType, TemperatureSensors
import psutil
import csv
import datetime
from pathlib import Path

metrics_path = Path("metrics.csv")
DELAY = 0.25

with NVMLLib() as lib:
    print("Driver Version:", lib.system.get_driver_version())
    if not metrics_path.exists():
        metrics_path.write_text("timestamp,index,name,gpu_memory_util,gpu_util,power,total_energy,temperature,gpu_memory_total,gpu_memory_reserved,gpu_memory_free,gpu_memory_used,cpu_util,host_memory_total,host_memory_used,host_memory_free,host_memory_percentage\n")
    with open(metrics_path, "a", newline="") as csvfile:
        for index, device in enumerate(lib.device):
            csv_writer = csv.writer(
                csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            while True:
                start=time.time()
                util = device.get_utilization_rates()
                gpu_memory = device.get_memory_info(version=2)
                host_memory =  psutil.virtual_memory()
                csv_writer.writerow(
                    [   
                        datetime.datetime.now().isoformat(),
                        index,
                        device.get_name(),
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
                        host_memory.percent
                    ]
                )
                end=time.time()
                time.sleep(DELAY - (end-start))

        # gpu_processes = device.get_graphics_running_processes()
        # processes = [Process(x.pid) for x in gpu_processes]
        # paths = [p.cmdline()[0] + "/" + p.name() for p in processes]
        # print(paths)
        # print([p.usedGpuMemory / 2**20 for p in gpu_processes])
