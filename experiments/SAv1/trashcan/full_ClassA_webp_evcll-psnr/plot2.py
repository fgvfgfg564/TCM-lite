import fire

from plotlib.rd import RD_plot, RDPlotData
from tools.utils import read_results, load_json


def main(sa_path: str, jpeg_path: str, bpg_path: str, vtm_path: str, output: str):
    data_sa = load_json(sa_path)
    results = dict()
    for target_bpp, data1 in data_sa.items():
        for target_time, data2 in data1.items():
            r = data2["avg_bpp"]
            d = data2["avg_psnr"]
            key = "Ours(" + target_time[12:] + "s)"
            results.setdefault(key, {"rd": [], "dashed": False, "color": None})
            results[key]["rd"].append((r, d))

    def detach_rd_t(rdt):
        rd = []
        t_sum = 0
        for r, d, t in rdt:
            rd.append((r, d))
            t_sum += t
        return rd, t_sum / len(rdt)

    data_jpeg = load_json(jpeg_path)
    results_jpeg = read_results(data_jpeg, load_t=True)
    rd, t_avg = detach_rd_t(results_jpeg)

    results[f"JPEG({t_avg:.1f}s)"] = {"rd": rd, "dashed": True, "color": None}

    data_bpg = load_json(bpg_path)
    results_bpg = read_results(data_bpg, load_t=True)
    rd, t_avg = detach_rd_t(results_bpg)
    results[f"BPG({t_avg:.1f}s)"] = {"rd": rd, "dashed": True, "color": None}

    # data_vtm = load_json(vtm_path)
    # results_vtm = read_results(data_vtm)
    # results["VTM"] = {
    #     "rd": results_vtm,
    #     "dashed": True,
    #     "color": None
    # }

    print(results)

    RD_plot(
        results,
        output,
        ylabel="PSNR",
        title="LIU4K(part)",
        xlim=(0.3, 1.0),
        ylim=(33, 41),
    )


if __name__ == "__main__":
    fire.Fire(main)
