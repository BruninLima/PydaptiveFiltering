# pydaptivefiltering/__init__.py

from .base import AdaptiveFilter
from .lms import *
from .rls import *
from .set_membership import *
from .lattice import *
from .fast_rls import *
from .qr_decomposition import *
from .iir import *
from .nonlinear import *

from .subband.cfdlms import CFD_LMS
from .subband.dlcllms import DLCL_LMS
from .subband.olsblms import OLSB_LMS

from .blind.Affine_Projection import Affine_Projection as Blind_Affine_Projection
from .blind.CMA import CMA
from .blind.Godard import Godard
from .blind.Sato import Sato

__version__ = "0.7.0"
__author__ = "BruninLima"

__all__ = ["AdaptiveFilter",
    "LMS", "NLMS", "AffineProjection", "SignData", "SignError", "DualSign", 
    "LMSNewton", "Power2ErrorLMS", "TDomainLMS", "TDomainDCT", "TDomainDFT",
    "RLS", "RLSAlt",
    "SMNLMS", "SMBNLMS", "SMAP", "Simplified_SMAP", "Simplified_PUAP",
    "LRLSPosteriori", "LRLSErrorFeedback", "LRLSPriori", "NormalizedLRLS",
    "FastRLS", "StabFastRLS",
    "QRRLS",
    "ErrorEquation", "GaussNewton", "GaussNewtonGradient", "RLSIIR", "SteiglitzMcBride",
    "BilinearRLS", "ComplexRBF", "MultilayerPerceptron", "RBF", "VolterraLMS", "VolterraRLS",
    "CFD_LMS", "DLCL_LMS", "OLSB_LMS", #todo subband
    "Blind_Affine_Projection", "CMA", "Godard", "Sato", #todo blind
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