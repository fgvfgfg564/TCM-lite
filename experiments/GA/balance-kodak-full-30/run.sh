BASEDIR=$(dirname "$0")
DATASET_DIR="/backup1/xyhang/dataset/kodak/*.png"

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 35 python tools/tester.py "$BASEDIR/EVC_LL" -i "$DATASET_DIR" --tools EVC --tool_filter EVC_LL -N 1000 --num-gen 100 --w_time 0 --ctu_size 256 --save_image --bpg_qp 30

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 35 python tools/tester.py "$BASEDIR/EVC_LM" -i "$DATASET_DIR" --tools EVC --tool_filter EVC_LM -N 1000 --num-gen 100 --w_time 0 --ctu_size 256 --save_image --bpg_qp 30

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 35 python tools/tester.py "$BASEDIR/EVC_LS" -i "$DATASET_DIR" --tools EVC --tool_filter EVC_LS -N 1000 --num-gen 100 --w_time 0 --ctu_size 256 --save_image --bpg_qp 30

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 35 python tools/tester.py "$BASEDIR/TCM" -i "$DATASET_DIR" --tools TCM -N 1000 --num-gen 100 --w_time 0 --ctu_size 256 --save_image --bpg_qp 30

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 35 python tools/tester.py "$BASEDIR/mixed" -i "$DATASET_DIR" --tools EVC TCM -N 1000 --num-gen 100 --w_time 0 0.25 0.5 0.75 1 1.25 1.5 2 --ctu_size 256 --save_image --bpg_qp 30