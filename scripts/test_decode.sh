#测试解码样例——TCM
python test/decode.py TCM IMAGE_FOLDER -o OUTPUT_PATH --code TCM_BIN_FOLDER

#测试解码样例——BPG
python test/decode.py BPG IMAGE_FOLDER --decoder-path BPG_DECODER_PATH --code BPG_BIN_FOLDER

#测试解码样例——VTM
python test/decode.py VTM IMAGE_FOLDER --build-dir VTM_BUILD_FOLDER --config VTM_CONFIG_FILE --code VTM_BIN_FOLDER