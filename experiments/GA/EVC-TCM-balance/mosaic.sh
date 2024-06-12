BASEDIR=$(dirname "$0")
DATASET_DIR="/backup1/xyhang/dataset/NIC_Dataset/test/ClassA_6K/IMG_3515.png"

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

# python tools/tester.py "$BASEDIR/mosaic" -i "$DATASET_DIR" --tools EVC TCM --tool_filter EVC_LS TCM_VBR2_2 -N 1000 --num-gen 1000 --w_time 0.029 --bpg_qp 32 --mosaic

python tools/illustrate.py -i experiments/EVC-TCM-balance/mosaic/results/1000/1000/32/0.029/0.05/False/0.2/256/IMG_3515.bin -o experiments/EVC-TCM-balance/mosaic/num_bytes.jpg --image "/backup1/xyhang/dataset/NIC_Dataset/test/ClassA_6K/IMG_3515.png" --type num_bytes --ctu_size 512 --mosaic

python tools/illustrate.py -i experiments/EVC-TCM-balance/mosaic/results/1000/1000/32/0.029/0.05/False/0.2/256/IMG_3515.bin -o experiments/EVC-TCM-balance/mosaic/method.jpg --image "/backup1/xyhang/dataset/NIC_Dataset/test/ClassA_6K/IMG_3515.png" --type method --ctu_size 512 --mosaic