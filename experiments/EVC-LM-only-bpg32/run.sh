BASEDIR=$(dirname "$0")
DATASET_DIR="/backup1/xyhang/dataset/NIC_Dataset/test/ClassA_6K/*.png"

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 33 python tools/tester.py "$BASEDIR" -i "$DATASET_DIR" --tools EVC --tool_filter EVC_LM -N 100 --num-gen 100