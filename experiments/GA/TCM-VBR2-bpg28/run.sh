BASEDIR=$(dirname "$0")
DATASET_DIR="/backup1/xyhang/dataset/NIC_Dataset/test/ClassA_6K/*.png"

taskset -a -c 17 python tools/tester.py "$BASEDIR" -i "$DATASET_DIR" --tools TCM --tool_filter TCM_VBR2_2 -N 100 --num-gen 100 --w_time 0