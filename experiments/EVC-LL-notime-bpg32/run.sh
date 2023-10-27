BASEDIR=$(dirname "$0")
DATASET_DIR="/backup1/xyhang/dataset/NIC_Dataset/test/ClassA_6K/*.png"


python tools/tester.py "$BASEDIR" -i "$DATASET_DIR" --tools EVC --tool_filter EVC_LL -N 100 --num-gen 100 --w_time 0