BASEDIR=$(dirname "$0")
DATASET_DIR=$BASEDIR/../../../images/6720x4480/IMG_3227.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 22 python -u tools/tester.py "$BASEDIR" -i "$DATASET_DIR" -a SAv1 --tools EVC WebP --tool_filter EVC_LL WebP --save_image --target_bpp 0.1 --ctu_size 512 --w_time 0.000 0.250 0.50 0.75 1.0 2.0 3.0 5.0 25.0 | tee ${BASEDIR}/main.log

sudo chown --recursive ${USER} ${BASEDIR}