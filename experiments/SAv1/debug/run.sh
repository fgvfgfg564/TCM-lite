BASEDIR=$(dirname "$0")
DATASET_DIR=$BASEDIR/../../../images/kodim23.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 22 python -u tools/tester.py "$BASEDIR" -i "$DATASET_DIR" -a SAv1 --tools EVC TCM --tool_filter EVC_LS EVC_LL TCM_VBR2_2 --save_image --target_bpp 0.25 --ctu_size 256 --w_time 0.0 0.5 1.0 2.0 | tee ${BASEDIR}/main.log

sudo chown --recursive ${USER} ${BASEDIR}