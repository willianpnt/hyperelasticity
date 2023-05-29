# -*- coding: utf-8 -*-
#
# Data structures for hyperelastic constitutive models
#
"""
HYPER Homework
Module 4: Hyperelasticity

WARNING:
in FiniteElement.update(), (S,M) = material.stress_stiffness(C)
--> stress_stiffness must be a function of right Cauchy Green tensor
===============================================================================
"""

import tensor
import numpy as np

class StVenantKirchhoffElasticity:
    """Data structure for (isotropic) St-Venant-Kirchhoff hyperelaticity models"""
    
    def __init__(self,E,nu):
        self.prop = dict()
        self.prop["YOUNG_MODULUS"] = E
        self.prop["POISSON_COEFFICIENT"] = nu
        self.prop["SHEAR_MODULUS"] = 0.5*E/(1.+nu)
        self.prop["BULK_MODULUS"] = E/3./(1.-2*nu)
        self.prop["1ST_LAME_CONSTANT"] = self.prop["BULK_MODULUS"]-2./3.*self.prop["SHEAR_MODULUS"]
        self.prop["2ND_LAME_CONSTANT"] = self.prop["SHEAR_MODULUS"]
    
    def getLame1(self,):
        return self.prop["1ST_LAME_CONSTANT"]
    
    def getLame2(self,):
        return self.prop["2ND_LAME_CONSTANT"]
    
    def potential(self,C):
        """Compute hyperelastic potential: phi = lambda/2 * tr(E)^2 - mu*(E:E)"""
        lam = self.getLame1()
        mu = self.getLame2()
        EL = 0.5*(C - tensor.I(len(C))) # Lagrangian strain E
        phi = lam/2.*(tensor.trace(EL))**2 + mu*np.tensordot(EL,EL,2)
        return phi

    def stress(self,C):
        """Compute 2nd Piola-Kirchhoff stress"""
        # =====================================================================
        PK2 = tensor.tensor(len(C)) 
        lam=self.getLame1()
        mu=self.getLame2()
        E=tensor.tensor(len(C))
        E=0.5*(C-tensor.I())
        PK2=lam*tensor.trace(E)*tensor.I()+2*mu*E
        # =====================================================================
        return PK2

    def stiffness(self,C):
        """Compute material tangent M = 2*dS/dC """
        # =====================================================================
        M = tensor.tensor4(len(C))
        lam=self.getLame1()
        mu=self.getLame2()
        M=lam*(tensor.outerProd4(tensor.I(),tensor.I()))+2*mu*tensor.IISym()
        # =====================================================================
        return M

    def stress_stiffness(self,C):
        """Compute 2nd Piola-Kirchhoff stress and material tangent at the same time"""
        # =====================================================================
        PK2 = tensor.tensor(len(C))
        M = tensor.tensor4(len(C))
        
        PK2 = self.stress(C)
        M = self.stiffness(C)
        # =====================================================================
        return (PK2,M)

class NeoHookeanElasticity:
    """Data structure for (isotropic) NeoHookean hyperelaticity models"""
    
    def __init__(self,E,nu):
        self.prop = dict()
        self.prop["YOUNG_MODULUS"] = E
        self.prop["POISSON_COEFFICIENT"] = nu
        self.prop["SHEAR_MODULUS"] = 0.5*E/(1.+nu)
        self.prop["BULK_MODULUS"] = E/3./(1.-2*nu)
        self.prop["1ST_LAME_CONSTANT"] = self.prop["BULK_MODULUS"]-2./3.*self.prop["SHEAR_MODULUS"]
        self.prop["2ND_LAME_CONSTANT"] = self.prop["SHEAR_MODULUS"]
    
    def getLame1(self,):
        return self.prop["1ST_LAME_CONSTANT"]
    
    def getLame2(self,):
        return self.prop["2ND_LAME_CONSTANT"]
    
    def potential(self,C):
        """Compute hyperelastic potential: phi = mu/2 * (tr(C)-3) - mu*ln(J) + lam/2 *ln(J)^2"""
        lam = self.getLame1()
        mu = self.getLame2()
        J = np.sqrt(tensor.det(C)) # J = det(F) and det(C) = J^2
        phi = mu/2.*(tensor.trace(C)-3.) - mu*np.log(J) + lam/2.*(np.log(J))**2.
        return phi

    def stress(self,C):
        """Compute 2nd Piola-Kirchhoff stress"""
        # =====================================================================
        PK2 = tensor.tensor(len(C))
        lam=self.getLame1()
        mu=self.getLame2()
        J=np.sqrt(tensor.det(C))
        PK2=mu*(tensor.I()-tensor.inv(C))+lam*(np.log(J))*tensor.inv(C)
        # =====================================================================
        return PK2

    def stiffness(self,C):
        """Compute material tangent M = 2*dS/dC """
        # =====================================================================
        M = tensor.tensor4(len(C))
        lam=self.getLame1()
        mu=self.getLame2()
        J=np.sqrt(tensor.det(C))
        Cdot=tensor.tensor4()
        Cdot=-0.5*(np.einsum('ik,jl->ijkl',tensor.inv(C),tensor.inv(C))\
                   +np.einsum('il,jk->ijkl',tensor.inv(C),tensor.inv(C)))
        M=lam*tensor.outerProd4(tensor.inv(C),tensor.inv(C))+2*(lam*np.log(J)-mu)*Cdot
        # =====================================================================
        return M

    def stress_stiffness(self,C):
        """Compute 2nd Piola-Kirchhoff stress and material tangent at the same time"""
        # =====================================================================
        PK2 = tensor.tensor(len(C))
        M = tensor.tensor4(len(C))
        PK2=self.stress(C)
        M=self.stiffness(C)
        
        # =====================================================================
        return (PK2,M)