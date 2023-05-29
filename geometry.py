# -*- coding: utf-8 -*-
#
# Elements geometries for finite element model
#

"""
HYPER Homework
Module 3: kinematic tensor field on a mesh
"""

import numpy as np



class SFT3:
    """Shape functions for 3-node triangle"""
    @staticmethod
    def shape(x):
        """Returns shape functions"""
        # =====================================================================
        N1 = 1-x[0]-x[1]
        N2 = x[0]
        N3 = x[1]
        return [N1,N2,N3]
        # =====================================================================

    @staticmethod
    def dShape(x):
        """Returns derivatives of shape functions (with respect to reference 
        coordinates)"""
        # =====================================================================
        dN1 = [-1, -1]
        dN2 = [1, 0]
        dN3 = [0, 1]
        return [dN1,dN2,dN3]
        # =====================================================================

class IPTri:
    """Numerical integration rules for triangles"""
    # 1-pt rule
    X = [[1./3.,1/3.]] # coordinates of integration point
    W =  [1./2.] # weight

class SFQ4:
    """Shape functions for 4-node quadrangle"""
    @staticmethod
    def shape(x):
        """Returns shape functions"""
        # =====================================================================
        N1 = 0.25*(1-x[0])*(1-x[1])
        N2 = 0.25*(1+x[0])*(1-x[1])
        N3 = 0.25*(1+x[0])*(1+x[1])
        N4 = 0.25*(1-x[0])*(1+x[1])
        return [N1,N2,N3,N4]
        # =====================================================================

    @staticmethod
    def dShape(x):
        """Returns derivatives of shape functions (with respect to reference 
        coordinates)"""
        # =====================================================================
        dN1 = [-0.25*(1-x[1]), -0.25*(1-x[0])]
        dN2 = [0.25*(1-x[1]),-0.25*(1+x[0])]
        dN3 = [0.25*(1+x[1]),0.25*(1+x[0])]
        dN4 = [-0.25*(1+x[1]),0.25*(1-x[0])]
        return [dN1,dN2,dN3,dN4]
        # =====================================================================

class IPGauss2D:
    """Numerical integration rules for quadrangles"""
    # 4-pt rule
    a = 1./np.sqrt(3)
    X =  [[-a,a], [-a,-a], [a,-a], [a,a]] # coordinates of integration points
    W = [1.0,1.0,1.0,1.0] # weights