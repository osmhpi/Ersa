"""This module loads and runs the gpu training jobs.
It is supposed to be invoked as a shell script in it's own subprocess."""
from pathlib import Path

import click
import dill  # pylint: disable=import-outside-toplevel
import torch  # pylint: disable=import-outside-toplevel


@click.command()
@click.argument("state")
@click.argument("cell")
@click.argument("model_name")
@click.argument("output")
def main(state, cell, model_name, output):
    """Run the training job.
    1. Load the state of the jupyter notebook from the provided file.
    2. Run the actual training script (cell).
    3. Save the model weights to the output path.
    """
    # load state from the users notebook
    dill.load_module(state)
    cell_lines = Path(cell).read_text(encoding="UTF-8")
    import megaclite.client

    megaclite.client.apply_pending_tensor_moves()
    megaclite.client.apply_pending_module_moves()
    # train the model
    exec(cell_lines)  # pylint: disable=exec-used
    # save the model
    torch.save(globals()[model_name].state_dict(), output)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
