BASEDIR=$(dirname "$0")
DATASET_DIR=$BASEDIR/../../images/kodim23.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 22 python tools/plot_precompute.py -o ${BASEDIR}/curves -i "$DATASET_DIR" --tools EVC TCM --tool_filter EVC_LS TCM_VBR2_2 --ctu_size 256 | tee ${BASEDIR}/main.log

sudo chown --recursive ${USER} ${BASEDIR}