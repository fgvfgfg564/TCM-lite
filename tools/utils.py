import json


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def read_results(data, d_key="avg_psnr", load_t=False):
    """
    Given JSON results, return a serie of R-D-T tuples in a list
    """

    def _read(o):
        results = []
        if isinstance(o, dict):
            if "avg_bpp" in o and d_key in o and "avg_t_dec" in o:
                r = o["avg_bpp"]
                d = o["avg_psnr"]
                if load_t:
                    t = o["avg_t_dec"]
                    return [(r, d, t)]
                else:
                    return [(r, d)]
            for v in o.values():
                results = results + _read(v)
        return results

    return _read(data)
