from IPython.core.magic import (Magics, magics_class, line_magic,
                                cell_magic, line_cell_magic, needs_local_scope)
from datetime import datetime
from IPython.core.display import display, HTML
from multiprocessing.connection import Client
import pickle
import dill
import torch
from pathlib import Path
import argparse
import shlex

@magics_class
class RemoteTrainingMagics(Magics):
    def __post_init__(self):
        self.host: str=None
        self.port: str=None
        self.key: str=None

    def send_training_job(self, cell: str, model, model_name: str):
        address = (self.host, self.port)
        conn = Client(address, authkey=self.key.encode())

        
        filename = f"state-{datetime.now().isoformat()}"
        dill.dump_module(filename)
        conn.send(filename)
        conn.send(cell)
        conn.send(model_name)

        stdout_listener = Client((self.host, 6002), authkey=b'secret password')
        while (message := stdout_listener.recv()) != "DONE":
            print(message, end="")
        
        weights_file = conn.recv()
        device = torch.device('cpu')
        model.load_state_dict(torch.load(weights_file, map_location=device))
        Path(filename).unlink()
        Path(weights_file).unlink()

    @line_magic
    def remote_config(self, line):
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument("host", type=str)
        parser.add_argument("port", type=int)
        parser.add_argument("key", type=str)

        args = parser.parse_args(shlex.split(line))

        self.host=args.host
        self.port=args.port
        self.key=args.key


    @cell_magic
    @needs_local_scope
    def train_remote(self, line, cell, local_ns):
        """Usage %%remote <model>"""
        if not line:
            raise TypeError("%%remote missing 1 required positional argument: 'model'")
        
        model_name = line.strip()
        display(HTML(f"<i>sending {model_name} to remote gpu</i>"))
        self.send_training_job(cell, local_ns[model_name], model_name)

    # @line_cell_magic
    # def lcmagic(self, line, cell=None):
    #     "Magic that works both as %lcmagic and as %%lcmagic"
    #     if cell is None:
    #         print("Called as line magic")
    #         return line
    #     else:
    #         print("Called as cell magic")
    #         return line, cell


def pre_run_hook(info):
    info.raw_cell


def load_ipython_extension(ipython):
    ipython.register_magics(RemoteTrainingMagics)
    # ipython.events.register("pre_run_cell", pre_run_hook)