python bin/encoder.py -i ~/home/dataset/kodak/kodim01.png --target-bpp 0 --model EVC_SS -o bin/test/kodim01.bin

python bin/decoder.py -i bin/test/kodim01.bin -o bin/test/kodim01.png