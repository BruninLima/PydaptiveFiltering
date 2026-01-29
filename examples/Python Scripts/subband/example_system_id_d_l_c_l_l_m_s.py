#################################################################################
#                        Example: System Identification                         #
#################################################################################
#                                                                               #
#  Subband algorithms can require filterbanks and/or block parameters.          #
#  We use the harness in pydaptivefiltering._utils.subband_id to handle:        #
#    - K multiple of L                                                          #      
#    - ensemble averaging                                                       #
#                                                                               #
#     Adaptive Algorithm used here: DLCLLMS                                 #
#                                                                               #         
#################################################################################

from __future__ import annotations

import numpy as np

import pydaptivefiltering as pdf
from pydaptivefiltering._utils.example_helper import (
    SubbandIDConfig,
    run_subband_system_id,
)
from pydaptivefiltering._utils.plotting import plot_learning_curve

def main(seed: int = 0, plot: bool = True):
    cfg = SubbandIDConfig(
        ensemble=50,
        K=4096,
        sigma_n2=0.001,
        Wo=np.array([0.32, -0.30, 0.50, 0.20], dtype=float),
    )


    out = run_subband_system_id(
        make_filter=lambda: pdf.DLCLLMS(n_subbands=64, filter_order=3, step_size=0.02, gamma=0.1, a=0.05),
        L=32,
        cfg=cfg,
        seed=seed,
        verbose_first=True,
    )

    if plot:
        plot_learning_curve(
            out["MSE_av"],
            out["MSEmin_av"],
            title="DLCLLMS learning curve (ensemble-averaged)",
        )

    return out
if __name__ == "__main__":
    main(seed=0, plot=True)
