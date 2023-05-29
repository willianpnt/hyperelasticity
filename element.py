# -*- coding: utf-8 -*-
#
# Data structures for finite element model
#

"""
HYPER Homework
Module 3: kinematic tensor field on a mesh
"""

import numpy as np
from scipy.linalg import polar
from scipy.linalg import logm

import tensor
from geometry import SFT3
from geometry import IPTri
from geometry import SFQ4
from geometry import IPGauss2D



class FiniteElement:
    """Data structure for finite element
    
    Attributes:
        - type: 2: Triangle, 3: Quadrangle
        - shape: shape functions N(Ksi)
        - dShape: derivatives of shape functions dN/dX = dN/dKsi*(dX/dKsi)^-1
        - weight: weight of the integration points w*det(dX/dKsi)
        - F: deformation gradient F = Grad(u) + I
        - hencky: Hencky strain ln(V) with F = V.R
        - E_GL: lagrangian strain E = 1/2 (C - I)
        - E_EA: Euler-Almansi strain e = 1/2 (I - b^-1)
        - PK1: piola kirchoff I stress P = F*S
        - sigma: cauchy stress
        - K: lagrangian tangent operator dP/dF
    """
    def __init__(self,t,xNod):
        self.type = t
        self.shape = []
        self.dShape = []
        self.weight = []
        #
        #--- select element type and relevant shape functions
        #
        if (t == 2): # triangle T3 (1 int. pt.)
            # =================================================================
            # on the example of the Q4 type below
            self.shape.append(SFT3.shape(IPTri.X[0]))
            dShape0 = np.array(SFT3.dShape(IPTri.X[0]))
            J = tensor.tensor(2)
            for n in range(3):
                J += tensor.outerProd(xNod[n][0:2],dShape0[n])
            self.dShape.append(np.dot(dShape0,tensor.inv(J)))
            self.weight.append(IPTri.W[0]*tensor.det(J))
            # =================================================================
        
        elif (t == 3): # quadrangle Q4 (4 int. pt.)
            for i in range(4): # loop on integration points
                self.shape.append(SFQ4.shape(IPGauss2D.X[i]))
                dShape0 = np.array(SFQ4.dShape(IPGauss2D.X[i])) # dN/dKsi 
                J = tensor.tensor(2) # dX/dKsi
                for n in range(4): # loop on nodes
                    J += tensor.outerProd(xNod[n][0:2],dShape0[n])
                self.dShape.append(np.dot(dShape0,tensor.inv(J)))
                self.weight.append(IPGauss2D.W[i]*tensor.det(J))
        #--- initialise mechanical tensors at each integration point
        #
        self.F = [] #deformation gradient F = Grad(u) + I
        self.hencky =[] #hencky strain ln(V) with F = V.R
        self.E_GL = [] # lagrangian strain E = 1/2 (C - I)
        self.E_EA = [] # E_EA strain e = 1/2 (I - b^-1)
        self.PK1 = [] # piola kirchoff I : P = F*S
        self.sigma = [] # cauchy stress : \sigma
        self.K = [] # lagrangian tangent operator dP/dF
        d = self.getdim() # space dimension
        for n in range(self.getnIntPts()):
            self.F.append(tensor.tensor(d))
            self.hencky.append(tensor.tensor(d))
            self.E_GL.append(tensor.tensor(d))
            self.E_EA.append(tensor.tensor(d))
            self.PK1.append(tensor.tensor(d))
            self.sigma.append(tensor.tensor(d))
            self.K.append(tensor.tensor4(d))

    def getdim(self,):
        """ Get mesh dimension """
        return np.shape(self.dShape)[0]

    def getnIntPts(self,):
        """ Get number of integration points per element """
        return len(self.shape)

    def getnNodes(self,):
        """ Get number of nodes per element """
        return np.shape(self.dShape)[1]

    def gradient(self,dShp,uNod):
        """
        Compute gradient of the displacement field

        Parameters
        ----------
        dShp : 2-entry array
            derivatives of shape functions.
        uNod : 2-entry array
            local displacement field.

        Returns
        -------
        G : 2-entry array
            gradient of the displacement field G_iJ=du_i/dX_J.

        """
        # =====================================================================
        # nodal field 'uNod' and the shape functions 'dShp'
        return (np.dot(np.transpose(uNod),dShp))
        # =====================================================================

    def update(self,uNod,mater):
        """ update strain and stress tensors of the element from the displacement
        field 'uNod' with the material model 'mater' """
        for i in range(np.shape(self.dShape)[0]): # loop on integration points
            # compute CG stretch tensor and GL strain tensor
            G = self.gradient(self.dShape[i],uNod)
            F = G+tensor.I(len(G))
            self.F[i] = F #store deformation gradient at integration point i
            C = tensor.rightCauchyGreen(F)
            self.E_GL[i] = 0.5*(C-tensor.I(len(C)))
            #compute spatial description
            _,V = polar(F,side='left') #from scipy.linalg.polar() method with u=R and p=V,
            V[V<1e-10]=1e-15 # replace pure zeros by very low values to prevent "nan" in np.log(V)
            self.hencky[i] = logm(V) # ln(V) with F = V.R, "true" strain
            b = tensor.leftCauchyGreen(F)
            self.E_EA[i] = 0.5*(tensor.I(len(b))-tensor.inv(b))
            ###
            if (mater == 0): #skip next lines: do not compute stress
                continue
            # compute PK2 stress tensor and material tangent operator M=2*dS/dC
            (PK2,M) = mater.stress_stiffness(C)
            # compute PK1 stress and lagrangian tangent operator K = dP/dF
            self.PK1[i] = tensor.PK2toPK1(F,PK2)
            self.K[i] = tensor.MaterialToLagrangian(F,PK2,M)
            # compute cauchy stress (spatial description)
            self.sigma[i] = tensor.PK1toCauchy(F,self.PK1[i])

    def computeForces(self,nNodes,dim):
        """
        compute internal forces of the element

        Parameters
        ----------
        nNodes : int
            number of nodes per element.
        dim : int
            space dimension.

        Returns
        -------
        fNod : 2-entry tensor
            internal nodal forces of the element.

        """
        
        fNod = np.zeros((nNodes,dim))
        
        
# =============================================================================
        p=self.getnIntPts()
        for i in range(p):
            wg=self.weight[i]
            P=self.PK1[i]
            G=self.dShape[i]
            fNod+=wg*np.dot(G,np.transpose(P))
# =============================================================================
        
        return fNod

    def computeStiffness(self,nNodes,dim):
        """
        compute internal stiffness of the element

        Parameters
        ----------
        nNodes : int
            number of nodes per element.
        dim : int
            space dimension.

        Returns
        -------
        KNod : 4-entry tensor
            internal stiffness of the element.

        """
        KNod = np.zeros((nNodes,dim,nNodes,dim))
# =============================================================================
        p=self.getnIntPts()
        
        for i in range(p):
            wg=self.weight[i]
            G=self.dShape[i]
            K=self.K[i]
            KNod+=wg*np.einsum('aJ,iJkL,bL->aibk',G,K,G)
# =============================================================================
        
        return KNod
