from typing import OrderedDict, Sequence, Tuple, TypedDict, Union
from typing_extensions import TypeAlias, Dict
import matplotlib.pyplot as plt

from .utils import savefig


class RDPlotDataItem(TypedDict):
    rd: Sequence[Tuple[float, float]]
    color: Union[str, None]
    dashed: bool


RDPlotData: TypeAlias = OrderedDict[str, RDPlotDataItem]


def RD_plot(
    data: RDPlotData, filename: str, ylabel: str, xlim=None, ylim=None, title=None
):
    """
    RDT plot; X axis is bpp and Y axis is PSNR; the radius of nodes indicate time. The last item in dict will be significant. Values should in R-D format.
    """

    plt.figure(figsize=(9, 6))
    for label, item in data.items():
        rd = item["rd"]
        r, d = tuple(zip(*rd))

        linestyle = "--" if item["dashed"] else "-"

        if item["color"] is not None:
            plt.plot(r, d, color=item["color"], linestyle=linestyle, label=label)
        else:
            plt.plot(r, d, linestyle=linestyle, label=label)

    plt.xlabel("BPP")
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if title:
        plt.title(title)
    savefig(data, filename)
