BASEDIR=$(dirname "$0")
DATASET_DIR=./images/gigapixel/*.jpg

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

python -u tools/tester.py "$BASEDIR" -i "$DATASET_DIR" -a SAv1 --tools WebP EVC --tool_filter WebP EVC_LL --save_image --target_bpp 0.4 0.6 0.8 1.0 --ctu_size 512 --target_time 30 45 60 90 --loss PSNR --num_steps 1000 | tee ${BASEDIR}/main.log