BASEDIR=$(dirname "$0")
export TYPE=num_bytes
export CTU_SIZE=512

export TYPE=method

export PYTHONPATH=.:..:coding_tools/MLIC/MLICPP

find ${BASEDIR}/results -type f -name "*.bin" | xargs -n 1 -P 16 -I {} sh -c 'python tools/illustrate.py -i {} --image "$(dirname "{}")"/"$(basename "{}" .bin)"_rec.bmp -o "$(dirname "{}")"/"$(basename "{}" .bin)"_${TYPE}.bmp --type ${TYPE} --ctu_size ${CTU_SIZE} --alpha 0.8'