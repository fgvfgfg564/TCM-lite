BASEDIR=$(dirname "$0")
DATASET_DIR=~/dataset/NIC_Dataset/test/ClassA_6K/*.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

export PYTHONPATH=.:..:coding_tools/MLIC/MLICPP

python -u tools/test_accelerator.py "$BASEDIR" -i "$DATASET_DIR" --tools EVC TCM MLICPP WebP JPEG --tool_filter EVC_LL TCM_VBR2_ALL MLICPP_ALL WebP JPEG --save_image --qscale 0.2 0.4 0.6 0.8 --ctu_size 256 --speedup 0.98 --loss PSNR --num_steps 1000 | tee ${BASEDIR}/main.log