BASEDIR=$(dirname "$0")
DATASET_DIR=~/dataset/NIC_Dataset/test/ClassA_6K/IMG_6726.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

export PYTHONPATH=.:..:coding_tools/MLIC/MLICPP

# python -u tools/test_accelerator.py "$BASEDIR" -i "$DATASET_DIR" --tools QARV EVC TCM MLICPP WebP JPEG --tool_filter QARV  EVC_LL TCM_VBR2_ALL MLICPP_ALL WebP JPEG --save_image --qscale 0.5 --ctu_size 512 --speedup 0.98 2 --loss PSNR --num_steps 1000 | tee ${BASEDIR}/main.log

python -u tools/tester.py "$BASEDIR" -i "$DATASET_DIR" -a QARV --save_image --quality 0.5 | tee ${BASEDIR}/anchor.log