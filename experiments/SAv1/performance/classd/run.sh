BASEDIR=$(dirname "$0")
DATASET_DIR=~/dataset/NIC_Dataset/test/ClassD_Kodak/*.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"
# 以QARV为baseline，时间相应缩放

export PYTHONPATH=.:..:coding_tools/MLIC/MLICPP

python -u tools/test_accelerator.py "$BASEDIR" -i "$DATASET_DIR" --tools QARV EVC TCM MLICPP --tool_filter QARV EVC_LL TCM_VBR2_ALL MLICPP_ALL WebP JPEG --qscale 0.3 0.4 0.5 0.6 0.7 --ctu_size 256 --speedup 0.01 0.25 0.50 0.75 1.00 1.25 1.50 2.0 3.0 --loss PSNR --num_steps 100 -o results.json | tee ${BASEDIR}/main.log