BASEDIR=$(dirname "$0")
DATASET_DIR=~/dataset/NIC_Dataset/test/ClassA_6K/DSC07099.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 22 python -u tools/tester.py "$BASEDIR" -i "$DATASET_DIR" -a SAv1 --tools WebP --tool_filter WebP --save_image --target_bpp 0.03 --ctu_size 512 --w_time 0.000 | tee ${BASEDIR}/main.log

sudo chown --recursive ${USER} ${BASEDIR}