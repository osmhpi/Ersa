FROM python:3.10.6
RUN apt update
RUN apt install -y libsqlite3-dev liblzma-dev libctypes-ocaml libreadline-dev libbz2-dev
RUN useradd -m --uid 9915 felix
ENV HOME="/home/felix"
USER felix
RUN curl https://pyenv.run | /bin/bash
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"
USER root
ADD . /app
RUN chown -R felix:felix /app
USER felix
ENV HOME="/root"
RUN python -m venv /app/venv
RUN /app/venv/bin/pip3 install -e /app
RUN /app/venv/bin/pip3 install git+https://github.com/sirexeclp/nvidia-ml-py3.git
CMD /bin/bash -c 'eval "$($HOME/.pyenv/bin/pyenv init -)" && source /app/venv/bin/activate && megaclite-server --socket /tmp/megaclite/sockets/socket0'