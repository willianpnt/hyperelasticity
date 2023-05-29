# -*- coding: utf-8 -*-
#
# Set of methods for manipulating 2nd-order and 4th-order tensors
#
"""
HYPER Homework
Module 1: operations on tensors
"""
#
# --- Namespace
#
import numpy as np #access to a method from the "numpy" library using "np.method()"
import numpy.linalg as la #access to a method from the "linalg" module of the
                          #"numpy" library using "la.method()"

#
# --- Vectors
#

def vector(d=2):
    """Constructor of a vector object (dimension d)"""
    return np.zeros(d)

#
# --- 2nd order tensors
#

def tensor(d=2):
    """Constructor of 2nd-order tensor (dimension d)"""
    return np.zeros((d,d))

def I(d=2):
    """Identity second-order tensor"""
    # =========================================================================
    return np.eye(d)
    # =========================================================================

def det(A):
    """Compute determinant of a matrix/tensor"""
    # =========================================================================
    return la.det(A)
    # =========================================================================

def inv(A):
    """Compute inverse of a matrix/tensor"""
    # =========================================================================
    return la.inv(A)
    # =========================================================================

def trace(A):
    """Compute trace of a matrix/tensor"""
    # =========================================================================
    return np.trace(A)
    # =========================================================================

def outerProd(a,b):
    """Compute outer product of two vectors"""
    # =========================================================================
    return np.outer(a,b)
    # =========================================================================

def rightCauchyGreen(F):
    """Compute right Cauchy-Green tensor from deformation gradient"""
    # =========================================================================
    C = np.dot(np.transpose(F),F)
    return C
    # =========================================================================

def leftCauchyGreen(F):
    """Compute left Cauchy-Green tensor from deformation gradient"""
    # =========================================================================
    b = np.dot(F,np.transpose(F))
    return b
    # =========================================================================

def PK2toPK1(F,S):
    """Compute Piola stress tensor from second Piola-Kirchhoff stress"""
    return np.dot(F,S)

def PK1toCauchy(F,P):
    """Compute Cauchy stress tensor from first Piola-Kirchhoff stress"""
    return (np.dot(P,np.transpose(F)) * 1/det(F))


#
# --- 4th order tensors
#

def tensor4(d=2):
    """Constructor of 4th-order tensor (dimension d)"""
    return np.zeros((d,d,d,d))

def II(d=2):
    """Identity fourth-order tensor"""
    # =========================================================================
    I = tensor4()
    for i in range(0,d):
        for j in range(0,d):
            for k in range(0,d):
                for l in range(0,d):
                    d_ik, d_jl = 0, 0
                    if i == k:
                        d_ik = 1
                    if j == l:
                        d_jl = 1
                    I[i,j,k,l] = d_ik * d_jl
    return I
    # =========================================================================

def IISym(d=2):
    """Symmetrical identity fourth-order tensor:
    IISym_ijkl = 1/2 * (delta_ik delta_jl + delta_il delta_jk)"""
    # =========================================================================
    Is = tensor4()
    for i in range(0,d):
        for j in range(0,d):
            for k in range(0,d):
                for l in range(0,d):
                    d_ik, d_jl, d_il,  d_jk  = 0, 0, 0, 0
                    if i == k:
                        d_ik = 1
                    if j == l:
                        d_jl = 1
                    if i == l:
                        d_il = 1
                    if j == k:
                        d_jk = 1
                    Is[i,j,k,l] = 0.5 * (d_ik * d_jl + d_il * d_jk)
    return Is
    # =========================================================================

def KK(d=2):
    """Spherical operator:
    returns the spherical part of a given 2nd-order tensor
    KK_ijkl = delta_ij * delta_kl"""
    # =========================================================================
    K = tensor4()
    for i in range(0,d):
        for j in range(0,d):
            for k in range(0,d):
                for l in range(0,d):
                    d_ij, d_kl= 0, 0
                    if i == j:
                        d_ij = 1
                    if k == l:
                        d_kl = 1
                    K[i,j,k,l] = d_ij * d_kl
    return K
    # =========================================================================

def outerProd4(a,b):
    """Compute outer product of two tensors"""
    # =========================================================================
    M = tensor4()
    for i in range(0,np.size(a,0)):
        for j in range(0,np.size(a,1)):
            for k in range(0,np.size(b,0)):
                for l in range(0,np.size(b,1)):
                    M[i,j,k,l] = a[i,j] * b[k,l]
    return M
    # =========================================================================

def MaterialToLagrangian(F,S,M):
    """Compute Lagrangian tangent operator from material tensor and stress"""
    l=np.size(S,0)
    Kron=np.eye(l)
    L=np.einsum('ik,JL->iJkL',Kron,S) + np.einsum('iI,IJKL,kK->iJkL',F,M,F)
    return L
