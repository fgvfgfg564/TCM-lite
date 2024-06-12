BASEDIR=$(dirname "$0")
DATASET_DIR=~/dataset/NIC_Dataset/test/ClassD_Kodak/*.png

export PYTHONPATH=.:..:coding_tools/MLIC/MLICPP

python -u tools/tester.py "$BASEDIR" -i "$DATASET_DIR" -a EVC --quality 0.0 0.05 0.1 0.15 0.2 0.25 0.5 0.75 1.0 --ctu_size 256 | tee ${BASEDIR}/main.log