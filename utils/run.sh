#! /bin/bash

# init pyenv
eval "$($HOME/.pyenv/bin/pyenv init -)"

# activate venv
source /venv/bin/activate

# start remotefs and ersa server
# python3 /remotefs/server.py /home/felix/projects/uni/phd/kisz/ersa/data \
# & 
exec python3 -m ersa.server -w 2 --socket /tmp/ersa/sockets/socket0