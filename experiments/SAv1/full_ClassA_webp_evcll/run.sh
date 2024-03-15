BASEDIR=$(dirname "$0")
DATASET_DIR=~/dataset/NIC_Dataset/test/ClassA_6K/*.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

python -u tools/tester.py "$BASEDIR" -i "$DATASET_DIR" -a SAv1 --tools WebP EVC --tool_filter WebP EVC_LL --save_image --target_bpp 0.03 0.06 0.12 0.18 --ctu_size 512 --w_time 0.000 0.250 0.50 0.75 1.0 25.0 | tee ${BASEDIR}/main.log