BASEDIR=$(dirname "$0")
DATASET_DIR=~/dataset/NIC_Dataset/test/ClassC_2K/*.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"
# 以QARV为baseline，时间相应缩放

export PYTHONPATH=.:..:coding_tools/MLIC/MLICPP

python -u tools/test_accelerator.py "$BASEDIR" -i "$DATASET_DIR" --tools QARV EVC TCM MLICPP WebP JPEG --tool_filter QARV EVC_LL TCM_VBR2_ALL MLICPP_ALL WebP JPEG --qscale 0.1 0.3 0.5 0.7 0.9 --speedup 0.01 0.50 0.75 --loss PSNR --num_steps 1000 -o results1.json | tee ${BASEDIR}/main1.log