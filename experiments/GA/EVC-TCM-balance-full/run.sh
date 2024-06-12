BASEDIR=$(dirname "$0")

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES sh -c "which python"

for filename in /backup1/xyhang/dataset/NIC_Dataset/test/ClassA_6K/*.png
do
    sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 6 python tools/tester.py "$BASEDIR/$(basename $filename)" -i "$filename" --tools EVC TCM --tool_filter EVC_LS TCM_VBR2_2 -N 1000 --num-gen 100 --w_time 0.000 0.012 0.029 0.050 0.080 0.125 100.0 --bpg_qp 30 31 32 33 34
done