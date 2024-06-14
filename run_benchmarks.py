import os
from pathlib import Path
import subprocess
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

RESULTS_DIR = Path("benchmark-results")

def run_notebook_locally(notebook_filename):
    print("running locally", notebook_filename)
    os.environ["MEGACLITE_DISABLE"] = "1"
    with open(notebook_filename) as file_ptr:
        notebook = nbformat.read(file_ptr, as_version=4)

    preprocessor = ExecutePreprocessor(timeout=600, kernel_name="python3")
    preprocessor.preprocess(notebook, {"metadata": {"path": "."}})

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    with open(RESULTS_DIR / Path(notebook_filename).name, 'w', encoding='utf-8') as file_ptr:
        nbformat.write(notebook, file_ptr)
    
    print("done")
    import time
    time.sleep(30)

def run_notebook_with_mega(notebook_filename):
    print("running with mega", notebook_filename)
    os.environ["MEGACLITE_DISABLE"] = "0"
    with open(notebook_filename) as file_ptr:
        notebook = nbformat.read(file_ptr, as_version=4)
    
    subprocess.run("docker compose up -d".split(" "))
    print("docker is up")

    preprocessor = ExecutePreprocessor(timeout=600, kernel_name="python3")
    preprocessor.preprocess(notebook, {"metadata": {"path": "."}})
    
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    with open(RESULTS_DIR / Path(notebook_filename).name, 'w', encoding='utf-8') as file_ptr:
        nbformat.write(notebook, file_ptr)
    
    subprocess.run("docker compose down".split(" "))

def main():
    repetitions = 1
    benchmark_dir = Path("benchmarks")
    for rep in range(repetitions):
        for benchmark in benchmark_dir.glob("*.ipynb"):
            run_notebook_locally(benchmark)
            run_notebook_with_mega(benchmark)


if __name__ == "__main__":
    main()
