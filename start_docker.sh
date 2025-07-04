# !/user/bin/bash

USER=wphu
[[ -z ${1} ]] || USER=${1}

RANDOM_NUM=$$$(date +%s)
NAME=${USER}_${RANDOM_NUM}

# DOCKER_IMAGE=harbor.aibee.cn/auto_car/visualglm:llava.v1.5  # cuda117+bitsandbytes=0.41.0
# DOCKER_IMAGE=harbor.aibee.cn/auto_car/visualglm:bizevo.v1.0  # transformers=4.44.0 + langchain + movipy cuda11.6
#DOCKER_IMAGE=harbor.aibee.cn/auto_car/visualglm:bizevo.v1.0.mmdet  # transformers=4.44.0 + langchain + movipy cuda11.6, mmdet, mmcv
#DOCKER_IMAGE=harbor.aibee.cn/auto_car/visualglm:bizevo.v1.1  # transformers=4.44.0 + langchain + movipy cuda11.6 + label_studio
#DOCKER_IMAGE=harbor.aibee.cn/auto_car/visualglm:llava_langchain.v1.5  # transformers=4.21.0 + langchain
# DOCKER_IMAGE=registry.aibee.cn/aibee/vlm-r1.torch2.5.1-cuda12.4-cudnn9-devel.v1.0
DOCKER_IMAGE=registry.aibee.cn/aibee/visualglm:bizevo.v1.0.agent_modelscope
docker pull ${DOCKER_IMAGE}

#CUDA_VISIBLE_DEVICES=6 \
# docker run --rm -it --gpus 6 \
nvidia-docker run --rm -it \
    -v /home/${USER}:/home/${USER} \
    -v /ssd/${USER}:/ssd/${USER} \
    -v /training/${USER}:/training/${USER} \
    -v /tracking/${USER}:/tracking/${USER} \
    -v ${PWD}/cache:/tmp/ \
    -v ${PWD}:/workspace \
    --name ${NAME} \
    -p 7862:7860 \
    -p 8503:8501 \
    -p 6008:6006 \
    --shm-size 128G \
    ${DOCKER_IMAGE} bash -c "export PYTHONPATH=/workspace && bash"

    
