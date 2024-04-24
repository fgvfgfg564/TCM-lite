BASEDIR=$(dirname "$0")
DATASET_DIR=~/dataset/NIC_Dataset/test/ClassA_6K/*.png

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

python -u tools/tester.py "$BASEDIR" -i "$DATASET_DIR" -a BPG --save_image --qp 13 17 21 25 29 33 37 41 45 49 --level 1 3 5 7 8 9 | tee ${BASEDIR}/main.log