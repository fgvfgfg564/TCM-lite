import numpy as np
import scipy.interpolate
import json


def BD_PSNR(R1, PSNR1, R2, PSNR2, piecewise=1):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    PSNR1 = np.array(PSNR1)
    PSNR2 = np.array(PSNR2)

    p1 = np.polyfit(lR1, PSNR1, 3)
    p2 = np.polyfit(lR2, PSNR2, 3)

    # integration interval
    min_int = max(min(lR1), min(lR2))
    max_int = min(max(lR1), max(lR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(
            np.sort(lR1), PSNR1[np.argsort(lR1)], samples
        )
        v2 = scipy.interpolate.pchip_interpolate(
            np.sort(lR2), PSNR2[np.argsort(lR2)], samples
        )
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_diff = (int2 - int1) / (max_int - min_int)

    return avg_diff


def BD_RATE(
    R1, PSNR1, R2, PSNR2, piecewise=1, min_int=-float("inf"), max_int=float("inf")
):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2), min_int)
    max_int = min(max(PSNR1), max(PSNR2), max_int)

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(
            np.sort(PSNR1), lR1[np.argsort(PSNR1)], samples
        )
        v2 = scipy.interpolate.pchip_interpolate(
            np.sort(PSNR2), lR2[np.argsort(PSNR2)], samples
        )
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2 - int1) / (max_int - min_int)
    avg_diff = (np.exp(avg_exp_diff) - 1) * 100
    return avg_diff


import os

TIME_ANALYSIS_ENC = os.path.join(os.path.split(__file__)[0], "time_enc.tsv")
TIME_ANALYSIS_DEC = os.path.join(os.path.split(__file__)[0], "time_dec.tsv")


if __name__ == "__main__":
    cwd = os.path.split(__file__)[0]
    with open(os.path.join(cwd, "data.json"), "r") as f:
        data = json.load(f)

    bpp_baseline = data["cheng2020"]["bpp"]
    psnr_baseline = data["cheng2020"]["psnr"]
    for method in data.keys():
        bpp_ours = data[method]["bpp"]
        psnr_ours = data[method]["psnr"]
        time = data[method]["time"]
        print(
            "Method:",
            method,
            "\tBD-rate-RGB=",
            BD_RATE(bpp_baseline, psnr_baseline, bpp_ours, psnr_ours),
            "\ttime=",
            sum(time) / len(time),
        )
