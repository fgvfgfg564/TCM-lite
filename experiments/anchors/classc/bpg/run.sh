BASEDIR=$(dirname "$0")
DATASET_DIR=~/dataset/NIC_Dataset/test/ClassC_2K/*.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

export PYTHONPATH=.:..:coding_tools/MLIC/MLICPP

python -u tools/tester.py "$BASEDIR" -i "$DATASET_DIR" -a BPG --quality 13 17 21 25 29 33 37 41 45 49 | tee ${BASEDIR}/main.log