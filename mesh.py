# -*- coding: utf-8 -*-
#
# Data structures for unstructured mesh
#
"""
HYPER Homework
Module 2: nodal field on a mesh
"""

class Node:
    """Basic data structure for nodes"""
    def __init__(self,i,x,y,z):
        """Create node of label i and coordinates (x,y,z)"""
        self.id = i
        self.X = [x,y,z]
    def getID(self):
        return self.id
    def getX(self,i):
        return self.X[i]

class Element:
    """Basic data structure for elements"""
    def __init__(self,i,t,n):
        """Create element of label i, type t, with list of nodes n"""
        self.id = i
        self.type = t    # 2:Triangle, 3:Quadrangle
        self.nodes = []  # list of node label
        self.nodes.extend(n)
    def getID(self):
        return self.id
    def getType(self):
        return self.type
    def getNode(self,i):
        return self.nodes[i]
    def nNodes(self):
        """Return the number of nodes in the element"""
        return len(self.nodes)

class MeshData:
    """Class containing basic data structure for unstructured mesh"""
    def __init__(self,d=2):
        """Create mesh of dimension d"""
        self.dim = d # dimension of the mesh as an integer, 2 by default
        self.nodes = [] # list of all the nodes in the mesh as Node instances
        self.elems = [] # list of all the elements in the mesh as Element instances
    def getDimension(self):
        """Return dimension of the mesh"""
        # =====================================================================
        return self.dim
        # =====================================================================
    def addNode(self,i,x,y=0.,z=0.):
        self.nodes.append(Node(i,x,y,z))
    def getNode(self,i):
        """Return the node with label i"""
        # =====================================================================
        return self.nodes[i]
        # =====================================================================
    def nNodes(self):
        """Return the number of nodes in the mesh"""
        # =====================================================================
        return len(self.nodes)
        # =====================================================================
    def addElement(self,i,t,n):
        """Add element of label i, type t, nodes n to the mesh"""
        # =====================================================================
        self.elems.append(Element(i,t,n))
        return None
        # =====================================================================
    def getElement(self,i):
        """Return the element with label i"""
        # =====================================================================
        return self.elems[i]
        # =====================================================================
    def nElements(self):
        """Return the number of elements in the mesh"""
        # =====================================================================
        return len(self.elems)
        # =====================================================================