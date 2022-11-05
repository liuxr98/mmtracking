# liuxr25/track:env_mmtracking
docker run --rm -d --name mmtracking_env -p 45890:8888 -v $(pwd)/mmtracking:/workdir:z --shm-size 8G --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all liuxr25/track:env_mmtracking

# 615
docker run -d --name deepsort_yolov3 -p 19002:5000 --restart=always --shm-size 8G --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all liuxr25/track:deepsort_yolov3_cuda11.7
