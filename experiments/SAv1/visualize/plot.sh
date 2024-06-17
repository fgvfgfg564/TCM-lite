BASEDIR=$(dirname "$0")
export TYPE=num_bytes
export CTU_SIZE=512

export TYPE=method

export PYTHONPATH=.:..:coding_tools/MLIC/MLICPP

find ${BASEDIR} -type f -name "*.bin" | xargs -n 1 -P 16 -I {} sh -c 'python tools/illustrate.py -i {} --image images/6720x4480/$(basename "{}" .bin).png -o "$(dirname "{}")"/"$(basename "{}" .bin)"_${TYPE}.bmp --type ${TYPE} --ctu_size ${CTU_SIZE} --alpha 0.6'