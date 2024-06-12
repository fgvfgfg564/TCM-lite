BASEDIR=$(dirname "$0")
DATASET_DIR=~/dataset/NIC_Dataset/test/ClassA_6K/*.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

python -u tools/tester.py "$BASEDIR" -i "$DATASET_DIR" -a SAv1 --tools WebP EVC TCM --tool_filter WebP EVC_LL TCM_VBR2_ALL --target_bpp 0.4 0.6 0.8 1.0 --ctu_size 512 --target_time 1.0 2.0 5.0 10.0 30.0 --loss PSNR --num_steps 300 | tee ${BASEDIR}/main.log