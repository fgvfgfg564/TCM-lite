BASEDIR=$(dirname "$0")
DATASET_DIR=$BASEDIR/../../../images/kodim23.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

python -u tools/tester.py "$BASEDIR" -i "$DATASET_DIR" -a SAv1 --tools EVC TCM --tool_filter EVC_LS EVC_LL TCM_VBR2_2 --target_bpp 0.25 --ctu_size 256 --target_time 0.0 | tee ${BASEDIR}/main.log