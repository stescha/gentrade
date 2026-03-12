from dataclasses import dataclass

import numpy as np


@dataclass
class BtResult:
    buy_times: np.ndarray
    sell_times: np.ndarray
    values: np.ndarray
    positions: np.ndarray
    pnls: np.ndarray
