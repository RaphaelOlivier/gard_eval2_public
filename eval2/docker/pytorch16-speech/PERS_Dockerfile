########## PyTorch 1 Deep Speech with extra dependencies for speech enhancement and post processing #################

ARG armory_version
FROM twosixarmory/armory:${armory_version} AS armory-pytorch-deepspeech-base


ENV NUMBA_CACHE_DIR /tmp
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"

RUN /opt/conda/bin/conda install tensorflow-gpu \
 pytorch=1.6 \
 torchvision \
 cudatoolkit=10.1 -c pytorch && \
    /opt/conda/bin/conda clean --all

RUN /opt/conda/bin/conda install -c conda-forge librosa 
RUN /opt/conda/bin/pip install hydra-core==1.0.0rc1 \
  python-levenshtein \
  torchaudio \
  numba==0.43.0 \
  --ignore-installed llvmlite==0.32.1 \
  soundfile \
  sox \
  warpctc-pytorch==0.2.1+torch16.cuda100
########## PyTorch 1 Deep Speech Release #################

FROM armory-pytorch-deepspeech-base as armory-pytorch-deepspeech
ARG armory_version
RUN /opt/conda/bin/pip install --no-cache-dir \ 
    armory-testbed==${armory_version} \
    adversarial-robustness-toolbox==1.4.2
CMD tail -f /dev/null

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

#WORKDIR /tmp
#RUN  git clone https://github.com/SeanNaren/warp-ctc.git
#RUN apt-get update && apt-get -y install cmake protobuf-compiler
#WORKDIR /tmp/warp-ctc
#ENV CUDA_HOME="/usr/local/cuda"
#RUN nvcc -V
#RUN  mkdir build
#WORKDIR /tmp/warp-ctc/build  
#RUN cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda .. && make
#WORKDIR /tmp/warp-ctc/pytorch_binding
#RUN python setup.py install

WORKDIR /workspace