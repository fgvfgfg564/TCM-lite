BASEDIR=$(dirname "$0")
DATASET_DIR="/backup1/xyhang/dataset/kodak/kodim01.png"

python tools/tester.py "$BASEDIR" -i "$DATASET_DIR" --tools TCM --tool_filter TCM_VBR2_2 -N 100 --num-gen 100 --bpg_qp 26