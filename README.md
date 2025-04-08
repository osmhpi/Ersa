[![hpi_logo.png](images/hpi_logo.png)](https://hpi.de/en/research/hpi-data-center/ai-service-center/)

# Ersa (Jupyter Remote GPU)

This project provides an ipython magic extension, to enable training pytorch models on remote GPU servers.

This project is named `ersa` after one of the moons of Jupiter.[1]

[1]: https://en.wikipedia.org/wiki/Ersa_(moon)


## Getting Started

~~~bash
# clone the repo
git clone git@github.com:osmhpi/ersa.git
cd ersa
# create a new virtual environment
python3 -m venv venv
source ./venv/bin/activate
# install dependencies and additional dependencies for the demo
pip3 install .[demo]
~~~

Forward port 6001 to a remote server with a running server instance.

~~~bash
ssh -L 6001:127.0.0.1:6001 my-gpu-server.com
~~~

You should now be able to run the `client.ipynb` notebook.

The first run will probably be quite slow, because the server will recreate your local venv, including compiling the correct python version.

## Server Setup

You need pyenv installed.
make sure to have the following packages installed

- libsqlite3-dev
- libsqlite3-dev liblzma-dev libctypes-ocaml libreadline-dev libbz2-dev 

libffi-dev ?
libssl-dev ?

---
![BMBF_non_transparent.jpeg](images/BMBF_non_transparent.jpeg) 

This is an artefact from KI-Servicezentrum Berlin-Brandenburg. KI-Servicezentrum Berlin-Brandenburg is funded by the Federal Ministry of Education and Research under the funding code 01IS22092.