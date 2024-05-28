BASEDIR=$(dirname "$0")
DATASET_DIR=./images/gigapixel/*.jpg

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

python -u tools/tester.py "$BASEDIR" -i "$DATASET_DIR" -a SAv1 --tools WebP EVC --tool_filter WebP EVC_LL --save_image --target_bpp 1.5 2.0 2.5 3.0 --ctu_size 512 --target_time 15 30 60 90 --loss PSNR --num_steps 10000 | tee ${BASEDIR}/main.log