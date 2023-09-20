from TCM.bin.engine import TCMModelEngine


encoder = TCMModelEngine('encode')
decoder = TCMModelEngine('decode')

weight_path = "/home/zqge/VCIP_zqge/vcip_vbr_best.pth.tar"

encoder.load(weight_path)
decoder.load(weight_path)