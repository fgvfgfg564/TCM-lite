BASEDIR=$(dirname "$0")
DATASET_DIR=~/dataset/kodak/*.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

export PYTHONPATH=.:..:coding_tools/MLIC/MLICPP

python -u tools/test_accelerator.py "$BASEDIR" -i "$DATASET_DIR" --tools TCM WebP JPEG --tool_filter TCM_VBR2_ALL --save_image --target_bpp 0.4 0.6 0.8 1.0 --ctu_size 256 --speedup 1.5 2.0 3.0 6.0 --loss PSNR --num_steps 100 | tee ${BASEDIR}/main.log