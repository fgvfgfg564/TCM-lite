BASEDIR=$(dirname "$0")
DATASET_DIR=~/dataset/NIC_Dataset/test/ClassA_6K/*.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

python -u tools/tester.py "$BASEDIR" -i "$DATASET_DIR" -a SAv1 --tools WebP EVC --tool_filter WebP EVC_LL --save_image --target_bpp 0.4 0.6 0.8 1.0 --ctu_size 512 --target_time 0.5 1.0 1.5 2.0 2.5 --loss MS-SSIM --num_steps 300 | tee ${BASEDIR}/main.log