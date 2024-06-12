BASEDIR=$(dirname "$0")
DATASET_DIR="/backup1/xyhang/dataset/NIC_Dataset/test/ClassA_6K/DOG_4507.png"

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 34 python tools/tester.py "$BASEDIR/1" -i "$DATASET_DIR" --tools EVC TCM --tool_filter EVC_LS TCM_VBR2_2 -N 1000 --num-gen 1000 --w_time 0 --bpg_qp 30

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 34 python tools/tester.py "$BASEDIR/2" -i "$DATASET_DIR" --tools EVC TCM --tool_filter EVC_LS EVC_LS TCM_VBR2_2 -N 1000 --num-gen 1000 --w_time 0 --bpg_qp 30

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 34 python tools/tester.py "$BASEDIR/3" -i "$DATASET_DIR" --tools EVC TCM --tool_filter EVC_LS EVC_LS EVC_LS TCM_VBR2_2 -N 1000 --num-gen 1000 --w_time 0 --bpg_qp 30

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 34 python tools/tester.py "$BASEDIR/4" -i "$DATASET_DIR" --tools EVC TCM --tool_filter EVC_LS EVC_LS EVC_LS EVC_LS TCM_VBR2_2 -N 1000 --num-gen 1000 --w_time 0 --bpg_qp 30

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 34 python tools/tester.py "$BASEDIR/5" -i "$DATASET_DIR" --tools EVC TCM --tool_filter EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS TCM_VBR2_2 -N 1000 --num-gen 1000 --w_time 0 --bpg_qp 30

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 34 python tools/tester.py "$BASEDIR/6" -i "$DATASET_DIR" --tools EVC TCM --tool_filter EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS TCM_VBR2_2 -N 1000 --num-gen 1000 --w_time 0 --bpg_qp 30

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 34 python tools/tester.py "$BASEDIR/7" -i "$DATASET_DIR" --tools EVC TCM --tool_filter EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS TCM_VBR2_2 -N 1000 --num-gen 1000 --w_time 0 --bpg_qp 30

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 34 python tools/tester.py "$BASEDIR/8" -i "$DATASET_DIR" --tools EVC TCM --tool_filter EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS TCM_VBR2_2 -N 1000 --num-gen 1000 --w_time 0 --bpg_qp 30

sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 34 python tools/tester.py "$BASEDIR/9" -i "$DATASET_DIR" --tools EVC TCM --tool_filter EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS TCM_VBR2_2 -N 1000 --num-gen 1000 --w_time 0 --bpg_qp 30

# sudo --preserve-env=PATH,PYTHONPATH,CUDA_VISIBLE_DEVICES nice -n -20 taskset -a -c 34 python tools/tester.py "$BASEDIR/10" -i "$DATASET_DIR" --tools EVC TCM --tool_filter EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS EVC_LS TCM_VBR2_2 -N 1000 --num-gen 1000 --w_time 0 --bpg_qp 30