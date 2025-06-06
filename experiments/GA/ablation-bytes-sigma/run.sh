BASEDIR=$(dirname "$0")
DATASET_DIR="/backup1/xyhang/dataset/NIC_Dataset/test/ClassA_6K/DOG_4507.png"

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 16 python tools/tester.py "$BASEDIR" -i "$DATASET_DIR" --tools EVC TCM --tool_filter EVC_LS TCM_VBR2_2 -N 1000 --num-gen 100 --w_time 0.029 0.080 0.200 0.800 --bpg_qp 30 31 32 33 --bytes_sigma 2 8 32 128 256 512 1024 2048 4096