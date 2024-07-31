#! /bin/bash

# init pyenv
eval "$($HOME/.pyenv/bin/pyenv init -)"

# activate venv
source /venv/bin/activate

# start remotefs and megaclite server
# python3 /remotefs/server.py /home/felix/projects/uni/phd/kisz/megaclite/data \
# & 
exec python3 -m megaclite.server -w 2 --socket /tmp/megaclite/sockets/socket0