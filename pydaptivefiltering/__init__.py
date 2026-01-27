# pydaptivefiltering/__init__.py
from .main import AdaptiveFilter

from .LMS.LMS import LMS
from .LMS.NLMS import NLMS
from .LMS.AffineProjection import AffineProjection
from .LMS.SignData import SignData
from .LMS.SignError import SignError
from .LMS.DualSign import DualSign
from .LMS.LMS_Newton import LMS_Newton
from .LMS.Power2_Error import Power2_Error
from .LMS.TDomain import TDomain
from .LMS.TDomain_DCT import TDomain_DCT
from .LMS.TDomain_DFT import TDomain_DFT

from .RLS.RLS import RLS
from .RLS.RLS_Alt import RLS_Alt

from .SetMembership.NLMS import SM_NLMS
from .SetMembership.BNLMS import SM_BNLMS
from .SetMembership.AP import SM_AP
from .SetMembership.Simp_AP import SM_Simp_AP
from .SetMembership.Simp_PUAP import SM_Simp_PUAP

from .LatticeRLS.LRLS_pos import LatticeRLS
from .LatticeRLS.LRLS_EF import LatticeRLSErrorFeedback
from .LatticeRLS.LRLS_priori import LatticeRLS_Priori
from .LatticeRLS.NLRLS_pos import NormalizedLatticeRLS

from .Fast_Transversal_RLS.Fast_RLS import FastRLS
from .Fast_Transversal_RLS.Stab_Fast_RLS import StabFastRLS

from .QR.RLS import QR_RLS

from .IIR_Filters.ErrorEquation import ErrorEquation
from .IIR_Filters.GaussNewton import GaussNewton
from .IIR_Filters.GaussNewton_GradientBased import GaussNewton_GradientBased
from .IIR_Filters.RLS_IIR import RLS_IIR
from .IIR_Filters.Steiglitz_McBride import Steiglitz_McBride

from .NonlinearFilters.Bilinear_RLS import Bilinear_RLS
from .NonlinearFilters.Complex_Radial_Basis_Function import Complex_Radial_Basis_Function
from .NonlinearFilters.Multilayer_Perceptron import Multilayer_Perceptron
from .NonlinearFilters.Radial_Basis_Function import Radial_Basis_Function
from .NonlinearFilters.Volterra_LMS import Volterra_LMS
from .NonlinearFilters.Volterra_RLS import Volterra_RLS

from .SubbandFilters.cfdlms import CFD_LMS
from .SubbandFilters.dlcllms import DLCL_LMS
from .SubbandFilters.olsblms import OLSB_LMS

from .BlindFilters.Affine_Projection import Affine_Projection as Blind_Affine_Projection
from .BlindFilters.CMA import CMA
from .BlindFilters.Godard import Godard
from .BlindFilters.Sato import Sato

__version__ = "0.2.0"
__author__ = "BruninLima"

__all__ = ["AdaptiveFilter",
    "LMS", "NLMS", "AffineProjection", "SignData", "SignError", "DualSign", 
    "LMS_Newton", "Power2_Error", "TDomain", "TDomain_DCT", "TDomain_DFT",
    "RLS", "RLS_Alt",
    "SM_NLMS", "SM_BNLMS", "SM_AP", "SM_Simp_AP", "SM_Simp_PUAP",
    "LatticeRLS", "LatticeRLSErrorFeedback", "LatticeRLS_Priori", "NormalizedLatticeRLS",
    "FastRLS", "StabFastRLS",
    "QR_RLS",
    "ErrorEquation", "GaussNewton", "GaussNewton_GradientBased", "RLS_IIR", "Steiglitz_McBride",
    "Bilinear_RLS", "Complex_Radial_Basis_Function", "Multilayer_Perceptron", "Radial_Basis_Function", "Volterra_LMS", "Volterra_RLS",
    "CFD_LMS", "DLCL_LMS", "OLSB_LMS",
    "Blind_Affine_Projection", "CMA", "Godard", "Sato",
    "info"]

def info():
    """Imprime informações sobre a cobertura de algoritmos da biblioteca."""
    print("\n" + "="*70)
    print("      PyDaptive Filtering - Complete Library Overview")
    print("      Reference: 'Adaptive Filtering' by Paulo S. R. Diniz")
    print("="*70)
    sections = {
        "Cap 3/4 (LMS)": "LMS, NLMS, Affine Projection, Sign Algorithms, Transform Domain",
        "Cap 5 (RLS)": "Standard RLS, Alternative RLS",
        "Cap 6 (Set-Membership)": "SM-NLMS, BNLMS, SM-AP, Simplified AP/PUAP",
        "Cap 7 (Lattice RLS)": "LRLS (Posteriori, Priori, Error Feedback), NLRLS",
        "Cap 8 (Fast RLS)": "Fast Transversal RLS, Stabilized FTRLS",
        "Cap 9 (QR)": "QR-Decomposition Based RLS",
        "Cap 10 (IIR)": "Error Equation, Gauss-Newton, Steinglitz-McBride, RLS-IIR",
        "Cap 11 (Nonlinear)": "Volterra (LMS/RLS), MLP, RBF, Bilinear RLS",
        "Cap 12 (Subband)": "CFDLMS, DLCLLMS, OLSBLMS",
        "Cap 13 (Blind)": "CMA, Godard, Sato, Blind Affine Projection"
    }
    for cap, algs in sections.items():
        print(f"\n{cap:25}: {algs}")
    
    print("\n" + "-"*70)
    print("Usage example: from pydaptivefiltering import LMS")
    print("Documentation: help(pydaptivefiltering.LMS)")
    print("="*70 + "\n")