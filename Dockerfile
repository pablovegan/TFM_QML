# Build container using ``docker build FOLDER_WITH_DOCKERFILE -t NAME_DOCKER``

# FROM python:3.8-slim-buster
FROM ubuntu

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get --yes -qq update \
    && apt-get --yes -qq upgrade \
    && apt-get --yes -qq install \
    bzip2 \
    cmake \
    cpio \
    curl \
    g++ \
    gcc \
    gfortran \
    git \
    gosu \
    libblas-dev \
    liblapack-dev \
    libopenmpi-dev \
    openmpi-bin \
    python3-dev \
    python3-pip \
    virtualenv \
    wget \
    zlib1g-dev \
    vim       \
    htop      \
    && apt-get --yes -qq clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", echo "Successfully built Docker image"]


# # For more information, please refer to https://aka.ms/vscode-docker-python
# # FROM python:3.8-slim
# FROM nersc/ubuntu-mpi:14.04


# RUN apt update
# RUN apt install python3
# RUN apt install python3-pip -y

# # Activate VirtualENV
# ENV PATH="/opt/venv/bin:$PATH"

# WORKDIR /app

# COPY requirements.txt /app
# RUN pip install -r requirements.txt

# COPY ./ /app

# CMD ["python3", echo "Successfully built Docker image"]



# # RUN apt-get update && apt-get install -y apt-utils --no-install-recommends
# RUN apt-get update

# RUN apt-get install -y tzdata \
#     build-essential \
#     tzdata \
#     ca-certificates \
#     ccache \
#     cmake \
#     curl \
#     git \
#     python3 \
#     python3-pip \
#     python3-venv \
#     libjpeg-dev \
#     libpng-dev && \
#     rm -rf /var/lib/apt/lists/* \
#     && /usr/sbin/update-ccache-symlinks \
#     && mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache \
#     && python3 -m venv /opt/venv
# # Activate VirtualENV
# ENV PATH="/opt/venv/bin:$PATH"

# # Keeps Python from generating .pyc files in the container
# ENV PYTHONDONTWRITEBYTECODE=1

# # Turns off buffering for easier container logging
# ENV PYTHONUNBUFFERED=1

# # Install pip requirements
# WORKDIR /qa_docker

# COPY . .
# RUN pip install wheel && pip install -r requirements.txt
# # RUN python -m pip install qubit-approximant

# # Creates a non-root user with an explicit UID and adds permission to access the /app folder
# # For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# # During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# CMD ["python"]
