BASEDIR=$(dirname "$0")
export TYPE=num_bytes
export CTU_SIZE=512

find ${BASEDIR}/results -type f -name "*.bin" | xargs -n 1 -P 16 -I {} sh -c 'python tools/illustrate.py -i {} --image "$(dirname "{}")"/"$(basename "{}" .bin)"_rec.png -o "$(dirname "{}")"/"$(basename "{}" .bin)"_${TYPE}.png --type ${TYPE} --ctu_size ${CTU_SIZE} --alpha 0.5'
export TYPE=method

find ${BASEDIR}/results -type f -name "*.bin" | xargs -n 1 -P 16 -I {} sh -c 'python tools/illustrate.py -i {} --image "$(dirname "{}")"/"$(basename "{}" .bin)"_rec.png -o "$(dirname "{}")"/"$(basename "{}" .bin)"_${TYPE}.png --type ${TYPE} --ctu_size ${CTU_SIZE} --alpha 0.5'

find ${BASEDIR}/results -type f -name "*.bin" | xargs -n 1 -P 16 -I {} sh -c 'python tools/illustrate.py -i {} --image "$(dirname "{}")"/"$(basename "{}" .bin)"_rec.png -o "$(dirname "{}")"/"$(basename "{}" .bin)"_mesh.png --type ${TYPE} --ctu_size ${CTU_SIZE} --alpha 0.'