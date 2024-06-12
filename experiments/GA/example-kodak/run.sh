BASEDIR=$(dirname "$0")
DATASET_DIR=$BASEDIR/../../../images/kodim23.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 22 python tools/tester.py "$BASEDIR" -i "$DATASET_DIR" --tools EVC TCM --tool_filter EVC_LS TCM_VBR2_2 -N 100 --num-gen 100 --w_time 0.00 0.25 0.50 0.75 1.0 1.25 1.50 --target_bpp 0.5 --ctu_size 256 | tee ${BASEDIR}/main.log

sudo chown --recursive ${USER} ${BASEDIR}