from app import TCMModelEngine
import os


encoder = TCMModelEngine('encode')
decoder = TCMModelEngine('decode')

weight_path = "/home/zqge/VCIP_zqge/vcip_vbr_best.pth.tar"

curdir = os.path.split(__file__)[0]

encoder.load(weight_path)
decoder.load(weight_path)

encoder.compile(os.path.join(curdir, "weights/debug_enc/"))
decoder.compile(os.path.join(curdir, "weights/debug_dec/"))