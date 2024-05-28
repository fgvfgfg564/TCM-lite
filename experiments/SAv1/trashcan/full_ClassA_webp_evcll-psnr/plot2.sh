BASEDIR=$(dirname "$0")

python -u ${BASEDIR}/plot2.py --output ${BASEDIR}/rd --sa_path ${BASEDIR}/results.json --jpeg_path experiments/anchors/jpeg-classA/results.json --bpg_path experiments/anchors/bpg-classA/results.json --vtm_path experiments/anchors/vtm-classA/results.json