BASEDIR=$(dirname "$0")
DATASET_DIR=~/dataset/NIC_Dataset/test/ClassD_Kodak/*.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

experiments/anchors/classb/bpg/run.sh

python -u tools/tester.py "$BASEDIR" -i "$DATASET_DIR" -a BPG --quality 13 17 21 25 29 33 37 41 45 49 | tee ${BASEDIR}/main.log