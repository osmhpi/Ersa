FROM python:3.10.6
RUN apt update
# pyenv dependencies
RUN apt install -y libsqlite3-dev liblzma-dev libctypes-ocaml libreadline-dev libbz2-dev
# fuse dependencies
RUN apt install -y libfuse3-dev fuse3
RUN useradd -m --uid 5087 felix
ENV HOME="/home/felix"
USER felix
RUN curl https://pyenv.run | /bin/bash
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"
USER root
ADD ./requirements.txt /app/
RUN chown -R felix:felix /app
RUN mkdir /venv
RUN chown -R felix:felix /venv
USER felix
RUN mkdir -p /home/felix/projects/uni/phd/kisz/megaclite/data
RUN python -m venv /venv
RUN /venv/bin/pip3 install -r /app/requirements.txt
RUN /venv/bin/pip3 install git+https://github.com/sirexeclp/nvidia-ml-py3.git
# ENV HOME="/root"
CMD /app/run.sh