BASEDIR=$(dirname "$0")
DATASET_DIR=$BASEDIR/../../images/6720x4480/*.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 21 python tools/tester.py "$BASEDIR" -i "$DATASET_DIR" --tools EVC TCM --tool_filter EVC_LS TCM_VBR2_2 -N 1000 --num-gen 100 --w_time 0.000 0.050 0.125 0.350 0.800 100.0 --save_image --bpg_qp 32 --ctu_size 512