#测试编码样例——TCM
python test/encode.py TCM IMAGE_FOLDER --bpg_path BPG_BIN_FOLDER --code TCM_BIN_FOLDER

#测试编码样例——BPG
python test/encode.py BPG IMAGE_FOLDER --encoder-path BPG_ENCODER_PATH --code BPG_BIN_FOLDER -j NUM_JOBS -q QUALITY

#测试编码样例——VTM
python test/encode.py VTM IMAGE_FOLDER -j NUM_JOBS -q QUALITY