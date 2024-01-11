import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import click


def run_notebook_locally(notebook_filename):
    os.environ["MEGACLITE_DISABLE"] = "1"
    with open(notebook_filename) as file_ptr:
        notebook = nbformat.read(file_ptr, as_version=4)

    preprocessor = ExecutePreprocessor(timeout=600, kernel_name="python3")
    output = preprocessor.preprocess(notebook, {"metadata": {"path": "."}})
    print(output)

def run_notebook_with_mega(notebook_filename):
    os.environ["MEGACLITE_DISABLE"] = "0"
    with open(notebook_filename) as file_ptr:
        notebook = nbformat.read(file_ptr, as_version=4)

    preprocessor = ExecutePreprocessor(timeout=600, kernel_name="python3")
    output = preprocessor.preprocess(notebook, {"metadata": {"path": "."}})
    print(output)


@click.argument("notebook_filename")
@click.command()
def main(notebook_filename):
    # run_notebook_locally(notebook_filename)
    run_notebook_with_mega(notebook_filename)


if __name__ == "__main__":
    main()
