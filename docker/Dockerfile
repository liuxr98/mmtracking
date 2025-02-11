ARG PYTORCH="1.9.0"
ARG CUDA="10.2"
ARG CUDNN="7"
ARG MM="2.8.0"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# 更换apt source.list
RUN rm /etc/apt/sources.list \
    && touch /etc/apt/sources.list \
    && echo "deb https://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb https://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb https://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb https://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb https://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse" >> /etc/apt/sources.list \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
    && apt-get update \
    && apt-get upgrade -y

RUN apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 wget\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9/index.html
RUN pip install mmdet

RUN conda clean --all
ENV FORCE_CUDA="1"

RUN git clone https://github.com/open-mmlab/mmtracking.git /mmtracking
WORKDIR /mmtracking
RUN pip install -r requirements/build.txt
RUN pip install -v -e .

RUN pip install git+https://github.com/JonathonLuiten/TrackEval.git

RUN conda install jupyter -y -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
    && conda install nbconvert==5.4.1 -y -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
    && jupyter notebook --generate-config \
    && echo "c.NotebookApp.allow_remote_access = True" >> ~/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.ip='*'" >> ~/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.password=''" >> ~/.jupyter/jupyter_notebook_config.py \
    && conda clean -a -y
    
RUN pip install --no-cache-dir jupyter_contrib_nbextensions -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && jupyter contrib nbextension install --user
    
RUN mkdir -p /workdir
WORKDIR /workdir

CMD ["jupyter","notebook","--no-browser","--allow-root","--NotebookApp.token=''"]