BASEDIR=$(dirname "$0")
DATASET_DIR=~/dataset/8K_test_single_image/*.jpg

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

export PYTHONPATH=.:..:coding_tools/MLIC/MLICPP

python -u tools/tester.py "$BASEDIR" -i "$DATASET_DIR" -a WebP --quality 0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 | tee ${BASEDIR}/main.log