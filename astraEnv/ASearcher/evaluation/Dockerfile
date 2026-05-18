ARG BASE_IMAGE

FROM ${BASE_IMAGE} as searchr1

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Install base packages
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

ARG CONDA_ENV_NAME=retriever
ARG PYTHON_VERSION=3.10
RUN conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y && \
    conda install -n $CONDA_ENV_NAME -c pytorch -c nvidia \
        pytorch=2.4.0 torchvision=0.19.0 torchaudio=2.4.0 pytorch-cuda=12.1 -y && \
    conda run -n $CONDA_ENV_NAME pip install transformers datasets pyserini

RUN conda install -n $CONDA_ENV_NAME -c pytorch -c nvidia faiss-gpu=1.8.0 -y
RUN conda run -n $CONDA_ENV_NAME pip install FlagEmbedding && \
    conda run -n $CONDA_ENV_NAME pip install uvicorn fastapi

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "retriever", "bash"]