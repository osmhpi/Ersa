import nbformat
import click
from pathlib import Path

@click.command()
@click.argument("notebook")
def main(notebook):
    notebook = Path(notebook)
    
    if not notebook.exists():
        raise FileNotFoundError(str(notebook))

    with open(notebook, "r") as file:
        nb_corrupted = nbformat.reader.read(file)
    
    notebook.rename(notebook.with_suffix(".bck.ipynb"))

    nb_fixed = nbformat.validator.normalize(nb_corrupted)
    
    with open(notebook, "w") as file:
        nbformat.write(nb_fixed[1], file)

if __name__ == "__main__":
    main()