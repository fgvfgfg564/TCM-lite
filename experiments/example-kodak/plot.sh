BASEDIR=$(dirname "$0")
export TYPE=num_bytes

find ${BASEDIR}/results -type f -name "*.bin" | xargs -n 1 -P 16 -I {} sh -c 'python tools/illustrate.py -i {} --image "$(dirname "{}")"/"$(basename "{}" .bin)"_rec.png -o "$(dirname "{}")"/"$(basename "{}" .bin)"_${TYPE}.png --type ${TYPE} --ctu_size 256 --alpha 1.'