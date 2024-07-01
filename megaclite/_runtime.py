"""This module loads and runs the gpu training jobs.
It is supposed to be invoked as a shell script in it's own subprocess."""
from pathlib import Path
import dill  # pylint: disable=import-outside-toplevel
import sys

"""Run the training job.
1. Load the state of the jupyter notebook from the provided file.
2. Run the actual training script (cell).
3. Save the model weights to the output path.
"""
_, state, cell, output = sys.argv
# load state from the users notebook
dill.load_module(state)
cell_lines = Path(cell).read_text(encoding="UTF-8")

apply_pending_tensor_moves()
apply_pending_module_moves()
# train the model
exec(cell_lines)  # pylint: disable=exec-used
reverse_module_moves()
reverse_tensor_moves()
# save the state
dill.dump_module(output)