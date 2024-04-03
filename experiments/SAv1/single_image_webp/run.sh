BASEDIR=$(dirname "$0")
DATASET_DIR=~/dataset/NIC_Dataset/test/ClassA_6K/DSC07099.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

python -u tools/tester.py "$BASEDIR" -i "$DATASET_DIR" -a SAv1 --tools WebP EVC --tool_filter WebP EVC_LL --save_image --target_bpp 0.30 --target_time 1.0 2.0 3.0 --ctu_size 512 | tee ${BASEDIR}/main.log