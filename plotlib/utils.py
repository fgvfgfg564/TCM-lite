import json
from typing import Any

import matplotlib.pyplot as plt


def savefig(data: Any, filename: str):
    plt.savefig(filename + ".png", dpi=300)
    plt.savefig(filename + ".pdf")
    with open(filename + ".json", "w") as f:
        json.dump(data, f, indent="\t", sort_keys=True)
    plt.close()
