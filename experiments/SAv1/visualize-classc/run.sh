BASEDIR=$(dirname "$0")
DATASET_DIR=images/ClassC-selected/*.png
SAMPLE_IMG=images/ClassC-selected/DSC08723.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"
# 以QARV为baseline，时间相应缩放

export PYTHONPATH=.:..:coding_tools/MLIC/MLICPP

python -u tools/test_ablation.py "$BASEDIR" -i "$DATASET_DIR" --tools QARV EVC TCM MLICPP WebP JPEG --tool_filter QARV EVC_LL TCM_VBR2_ALL MLICPP_ALL WebP JPEG --ctu_size 512 --time_limits 300 --levels 6 --qscale 0.3 -o results_visualize.json | tee ${BASEDIR}/across_imafes.log

# python -u tools/test_ablation.py "$BASEDIR" -i "$SAMPLE_IMG" --tools QARV EVC TCM MLICPP WebP JPEG --tool_filter QARV EVC_LL TCM_VBR2_ALL MLICPP_ALL WebP JPEG --speedup 0.01 0.25 0.5 0.75 1.0 1.25 1.5 2.0 3.0 --ctu_size 512 --time_limits 300 --levels 6 --qscale 0.3 -o results_single_img.json | tee ${BASEDIR}/single_image_different_speed.log