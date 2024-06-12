BASEDIR=$(dirname "$0")
DATASET_DIR=~/dataset/NIC_Dataset/test/ClassA_6K/DSC_3889.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"
# 以QARV为baseline，时间相应缩放

export PYTHONPATH=.:..:coding_tools/MLIC/MLICPP

python -u tools/test_accelerator.py "$BASEDIR" -i "$DATASET_DIR" --tools EVC QARV TCM MLICPP WebP JPEG --tool_filter EVC_LL QARV TCM_VBR2_ALL MLICPP_ALL WebP JPEG --save_image --qscale 0.3 --ctu_size 512 --speedup 0.99 --loss PSNR --num_steps 100 -o results3.json | tee ${BASEDIR}/main3.log