BASEDIR=$(dirname "$0")
DATASET_DIR=~/dataset/kodak/*.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

export PYTHONPATH=.:..:coding_tools/MLIC/MLICPP

python -u tools/test_accelerator.py "$BASEDIR" -i "$DATASET_DIR" --tools MLICPP WebP JPEG --tool_filter MLICPP_ALL WebP JPEG --save_image --qscale 0.2 0.4 0.6 0.8 --ctu_size 256 --speedup 0.98 1.5 2.0 3.0 5.0 --loss PSNR --num_steps 100 | tee ${BASEDIR}/main.log