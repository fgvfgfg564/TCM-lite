BASEDIR=$(dirname "$0")
DATASET_DIR=images/6720x4480/IMG_6726.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"
# 以QARV为baseline，时间相应缩放

export PYTHONPATH=.:..:coding_tools/MLIC/MLICPP

python -u tools/test_ablation.py "$BASEDIR" -i "$DATASET_DIR" --tools QARV EVC TCM MLICPP WebP JPEG --tool_filter QARV EVC_LL TCM_VBR2_ALL MLICPP_ALL WebP JPEG --save_image --ctu_size 512 --time_limits 120 300 600 -o results_ablation.json | tee ${BASEDIR}/main1.log