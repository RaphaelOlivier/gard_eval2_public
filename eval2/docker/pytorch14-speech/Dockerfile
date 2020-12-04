########## PyTorch 1 Deep Speech with extra dependencies for speech enhancement and post processing #################

ARG armory_version
FROM twosixarmory/pytorch-deepspeech:${armory_version} AS armory-pytorch-deepspeech

########## Extra packages for defense #############

WORKDIR /
RUN  git clone https://github.com/RaphaelOlivier/SCTK.git
WORKDIR /SCTK
RUN make config && make all && make check && make install && make doc
WORKDIR /tmp
RUN git clone --recursive https://github.com/parlance/ctcdecode.git
WORKDIR /tmp/ctcdecode 
RUN pip install .
WORKDIR /tmp
RUN git clone https://github.com/RaphaelOlivier/pyaudlib.git --branch gard
WORKDIR /tmp/pyaudlib
RUN pip install . 
WORKDIR /tmp
RUN pip install tensorboardX
RUN git clone https://github.com/santi-pdp/ahoproc_tools.git
WORKDIR /tmp/ahoproc_tools
RUN pip install .
WORKDIR /workspace