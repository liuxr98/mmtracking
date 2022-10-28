# liuxr25/track:env_mmtracking
docker run --rm -d --name mmtracking_env -p 45890:8888 -v $(pwd)/mmtracking:/workdir --shm-size 8G --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all liuxr25/track:env_mmtracking

