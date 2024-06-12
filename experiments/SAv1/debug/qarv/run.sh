BASEDIR=$(dirname "$0")
DATASET_DIR=images/kodim23.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

export PYTHONPATH=.:..:coding_tools/MLIC/MLICPP

python -u tools/tester.py "$BASEDIR" -i "$DATASET_DIR" -a SAv1 --tools QARV WebP --tool_filter QARV WebP --target_bpp 0.25 --ctu_size 256 --target_time 1000.0 --loss PSNR | tee ${BASEDIR}/main.log