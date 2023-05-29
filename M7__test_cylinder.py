# -*- coding: utf-8 -*-
#
# Tensile test on a membrane with hole, assuming plane strain conditions
#

"""
HYPER Homework
Module 7: non linear solver and 2D plane strain problem

Complete code below where indicated by
# ============
# TODO:
# ...
# ============

In this script, you must complete the non linear Newton-Raphson solver to compute
the results for a pressure test on a compressible hyperelastic thick cylinder,
in plane strain conditions (2D).
There is no external forces: the cylinder is loaded by means of prescribed radial
displacements on the interior surface.
(hence R = Tint and K = Kint as in previous homeworks)

You must test BOTH material models:
    - St Venant Kirchhoff
    - NeoHookean

You must COMPULSORILY test both triangular meshes:
    - cynlinder-tri_coarse.msh
    - cynlinder-tri_refine.msh

You can also test quandragular meshes:
    - cynlinder-quad_coarse.msh
    - cynlinder-quad_refine.msh

In the post-processing section, there is nothing to complete but you can modify
it as you wish.
It provides 2 figures, saved in .png files, showing the loading curve of the test
and the evolution of the displacement along the bottom line, to give a comparison
with the linear elastic result:
    u = u_r * e_r = a*r + b/r
with (a,b) constants which depend on the loading and the material properties.

"""
#
#--- namespace
#
import numpy as np
import matplotlib.pyplot as plt
import gmshtools
import fem
import elasticity
from datetime import datetime
import os
import errno
def mkdir_p(path):
    """ make directory at 'path' (full or relative) if not existing """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
print('======================================')
print('=== 2D plane strain thick cylinder ===')
print('======================================')

#######################
#%% problem definition
######################
print('*--------------------------*')
print('*--- PROBLEM DEFINITION ---*')
print('*--------------------------*')
# DEFAULT UNITS: m, Pa, N
#
#--- select material model (test both !)
#
# steel:
# Eyoung = 210e9 #Young's modulus (Pa)
# nu = 0.3 #Poisson's ratio
# matmodel = 'StVenant'
# # filled rubber (very stiff):
Eyoung = 10e6 #Young's modulus (Pa)
nu = 0.45 #Poisson's ratio
matmodel = 'NeoHookean'
dirout = os.path.join(matmodel) #directory of output files, named from chosen matrial model
print('Material model: '+matmodel+'(%.2e,%.2f)'%(Eyoung,nu))
#
#--- incremental loading parameters
#
nSteps = 5 #number of steps in the incremental loading loop
tStep = 1./nSteps #size of the steps
# rdisp = 0.0002 #constant radial displacement on interior boundary (StVenant)
rdisp = 0.01 #constant radial displacement on interior boundary (NeoHookean)
#
#--- solver parameters
#
PRECISION = 1e-15 # machine "zero"
TOLERANCE = 1e-6
ITERMAX = 10
verbosity = True # whether or not to print progression of the NR solver
#
#--- read geometry info from Gmsh file
#
with open('cylinder.geo') as geomfile:
    lines = geomfile.readlines()
    Ri = float(lines[0].strip().strip(";").split(" = ")[1]) #interior radius (m)
    Re = float(lines[1].strip().strip(";").split(" = ")[1]) #exterior radius (m)
print("Geometry: cylinder of radii Ri = %.2e m and Re = %.2e m"%(Ri,Re))
#
#--- read mesh from Gmsh file
#
inputfile = "cylinder-tri_coarse.msh"
# inputfile = "cylinder-tri_refined.msh"
# inputfile = "cylinder-quad_coarse.msh"
# inputfile = "cylinder-quad_refined.msh"
meshfile = open(os.path.join('msh',inputfile),'r')
mesh = gmshtools.gmshInput_mesh(meshfile,d=2)
dim = mesh.dim #dimension of the problem
nNodes = mesh.nNodes() #number of nodes in the mesh
nElements = mesh.nElements() #number of elements in the mesh
print("Read mesh '%s' with %d nodes and %d elements" % (inputfile,nNodes,nElements))
#
#--- gmsh output
#
mesh_name = inputfile.split('-')[0]
mesh_type = inputfile.split('-')[1][:-4] #'tri_coarse' = element type + mesh refinement
dirout = os.path.join(matmodel,mesh_type) #subdirectory for output files
mkdir_p(dirout) #create directory for output files if not already existing
outputfile = mesh_name + '-results_Nsteps-'+str(int(nSteps))+".msh"
outputfile = os.path.join(dirout,outputfile)
gmsh_out = open(outputfile,'w')
gmshtools.gmshOutput_mesh(gmsh_out,mesh)


##################
#%% preprocessing
#################
print('*----------------------*')
print('*--- PRE PROCESSING ---*')
print('*----------------------*')
#
#--- boundary conditions
#
# On the left side of the quarter-cylinder (X=0), the displacements in the x-direction are fixed (u_x =0)
# At the bottom of the quarter-cylinder (Y=0), the displacements in the y-direction are fixed (u_y=0)
# We apply a radial displacement "disp" on the interior of the cylinder, in the r-direction (u_r=rdisp)

nNodes = mesh.nNodes()
nDofs = dim*nNodes
print('Total number of degrees of freedom: %d' % nDofs)
fixDofs = [] #initialise empty list
# =============================================================================
# TODO: list degrees of freedom (d.o.f) in x-direction linked to left side (e.g. 
# with x<=Xmin) and d.o.f in y-direction linked to bottom side (e.g. with 
# y<=Ymin) in the list 'fixDofs'
# Hint: for loop on all the nodes and check coordinates with mesh.getNode(i).getX()


Xmin = min([mesh.getNode(i).getX(0) for i in range(mesh.nNodes())])
Ymin = min([mesh.getNode(i).getX(1) for i in range(mesh.nNodes())])


for i in range(mesh.nNodes()):
    if mesh.getNode(i).getX(0)==Xmin:
        fixDofs.append(i*mesh.dim)
    elif mesh.getNode(i).getX(1)==Ymin:
        fixDofs.append(i*mesh.dim+1)
        
        

# =============================================================================
# list nodes on the interior boundary, compute normal (e_r vector) and prescribe
# radial displacement u = rdisp * e_r
dispDofs = [] #initialise empty list
dispNodes = []
normals = []
for n in range(nNodes):
    X = mesh.getNode(n).getX(0)
    Y = mesh.getNode(n).getX(1)
    if (X**2+Y**2-Ri**2)<PRECISION: # X^2 + Y^2 = Ri^2
        dispNodes.append(n)
        dof = n*dim
        dispDofs.append(dof)
        dispDofs.append(dof+1)
        normals.append([X,Y])
# prescribe displacements according to the normal to the surface
normals = np.array(normals)/Ri # e_r = cos(theta), sin(theta) with cos(theta)=X/Ri and sin(theta)=Y/Ri
dispInt = rdisp * normals # applied to every node according normal to the surface in that node
disp = dispInt.flatten() # from 2-entry table to flattened array [u1,v1,u2,v2,...,uN,vN]
# d.o.f. subjected to a boundary condition:
bcDofs = fixDofs + dispDofs # the addition of 2 lists concatenates them
# d.o.f. free of any constraint (where the displacement field is to be determined):
listDofs = np.arange(nDofs)
freeDofs = np.delete(listDofs,bcDofs) # remove all entries in bcDofs from listDofs
#display fixed degrees of freedom
print('Fixed degrees of freedom: ['+', '.join(['{:d}'.format(fixdof) for fixdof in fixDofs])+']')

#
#--- create material model instance
#
if matmodel == "StVenant":
    material = elasticity.StVenantKirchhoffElasticity(Eyoung,nu)
elif matmodel == "NeoHookean":
    material = elasticity.NeoHookeanElasticity(Eyoung,nu)
else:
    raise(KeyError,"matmodel must be one of 'StVenant', 'NeoHookean'")

#
#--- create FE model instance
#
model = fem.FEModel(mesh,material)

#
#--- initialise displacement array
#
U = np.zeros(nDofs) #flattened vector of the 2-entry array U = [u1 v1 u2 v2 ... uN vN]

###############
#%% simulation
###############
print('*------------------*')
print('*--- SIMULATION ---*')
print('*------------------*')
start_time = datetime.now() #measure time for the simulation
time = 0.0 #final time is nSteps x tStep = nSteps x (1/nSteps) = 1
itermoy = 0. #average number of iterations per time step
displacements = [] #prescribed displacement per step on the right hand side of the membrane
reactions = [] #corresponding reaction per step

for iStep in range(nSteps): #loop on time steps --> incremental loading
    time += tStep
    print('--> Step %d (time = %.2f / disp = %.2e)' % (iStep+1,time,rdisp*time))

    #
    #--- non-linear (Newton-Raphson) solver
    #
    
    # initialisation
    # =========================================================================
    # TODO: apply Dirichlet boundary condition increment and compute initial
    # residual:
    # Hint: U[-- dof on interior boundary --] = disp*time
    # residual array: 
    for i in  range(len(dispNodes)):
        U[dispNodes[i]*2]=disp[i*2]*time
        U[dispNodes[i]*2+1]=disp[i*2+1]*time
        
    R = model.computeResidual(U.reshape((nNodes,dim)))
    R=R.reshape(nDofs)
    
    # =========================================================================
    
    norm = np.linalg.norm(R[np.ix_(freeDofs)])
    test = TOLERANCE*norm
    if (test < PRECISION): test = PRECISION
    iteration  = 0
    if verbosity: print('*** Iteration %02d: residual norm =%15.8e ***' % (iteration,norm))
    while (norm > test): #Newton-Raphson iterations

        # compute correction
        K = model.computeTangent(U.reshape((nNodes,dim)))
        K = K.reshape((nDofs,nDofs))
        # =====================================================================
        # TODO: compute dU by solving linearized system on free dofs
        # using K[np.ix_(freeDofs,freeDofs)] and R[np.ix_(freeDofs)]
        
        dU = -np.dot(np.linalg.inv(K[np.ix_(freeDofs,freeDofs)]),R[np.ix_(freeDofs)])
        
        # =====================================================================

        # stagnation criterion (on dU)
        normdU = np.linalg.norm(dU)
        if (normdU < PRECISION) :
            if verbosity: print("ERROR: no convergence in Newton loop, dQ stagnation")
            break

        # =====================================================================
        # TODO: update displacement
        
        for i in range(len(freeDofs)):
            U[freeDofs[i]] += dU[i]
        
        # =====================================================================

        # =====================================================================
        # TODO: update residual
        
        R=model.computeResidual(U.reshape((nNodes,dim)))
        R=R.reshape(nDofs)
        
        # =====================================================================

        # update residual norm
        norm = np.linalg.norm(R[np.ix_(freeDofs)])
        iteration = iteration+1
        if verbosity: print('*** Iteration %02d: residual norm =%15.8e ***' % (iteration,norm))

        # =====================================================================
        # TODO: maximum iteration criterion
        # stop Newton-Raphson iterations when ITERMAX is reached and print a message
        
        if iteration > ITERMAX:
            print('to much iterations')
            break
        
        # =====================================================================
    
    #end of Newton-Raphson iterations
    
    itermoy +=iteration
    if (norm <= test):
        if verbosity: print("COMPLETE: norm = %.5e < test = %.5e"%(norm,test))

    #
    #--- store prescribed displacement and corresponding reaction for current time step
    #
    nodalForce = [] ; uint = []
    for i in dispNodes:
        u = np.array(U.reshape(nNodes,dim)[i,:]).flatten()
        X = np.array([mesh.getNode(i).getX(0),mesh.getNode(i).getX(1)])
        x = X + u
        e_r = x/np.linalg.norm(x)
        nodalForce.append(R.reshape(nNodes,dim)[i,:].dot(e_r))
        uint.append(u.dot(e_r))
        assert np.allclose(rdisp*time,uint[-1])
    reactions.append(nodalForce) # (N)
    displacements.append(uint) # (m)
    #
    #--- get strain and stress tensor fields for current time step
    #
    E = []
    P = []
    hencky = []
    euler = []
    sigma = []
    for n in range(model.nElements()):
        E.append(model.getStrainGreenLagrange(n).flatten())
        P.append(model.getStressPK1(n).flatten())
        hencky.append(model.getStrainHencky(n).flatten())
        euler.append(model.getStrainEulerAlmansi(n).flatten())
        sigma.append(model.getStressCauchy(n).flatten())
    #
    #--- gmsh output for current time step
    #
    gmshtools.gmshOutput_nodal(gmsh_out,"Residual",R.reshape(nNodes,dim),iStep,time)
    gmshtools.gmshOutput_nodal(gmsh_out,"Displacements",U.reshape((nNodes,dim)),iStep,time)
    gmshtools.gmshOutput_element(gmsh_out,"Green-Lagrange strain",E,iStep,time)
    gmshtools.gmshOutput_element(gmsh_out,"Piola Kirchhoff I stress",P,iStep,time)
    gmshtools.gmshOutput_element(gmsh_out,"Euler-Almansi strain",euler,iStep,time)
    gmshtools.gmshOutput_element(gmsh_out,"Hencky strain",hencky,iStep,time)
    gmshtools.gmshOutput_element(gmsh_out,"Cauchy stress",sigma,iStep,time)

gmsh_out.close() # close gmsh output file
######################
#%% end of simulation
#####################
# print convergence status and solver output (iteration, CPU time)
print('####################################################')
if (norm<=test):
    print('############### SIMULATION COMPLETED ###############')
else:
    print('!SIMULATION FAILED: NO CONVERGENCE AT LAST TIME STEP ')
print('####################################################')
itermoy /= nSteps
print('Mean number of iteration per step : %.0f' % itermoy)
interval = datetime.now() - start_time #measure time for the simulation
print('Total time of computation:',str(interval))

####################
#%% post-processing
###################
print('*-----------------------*')
print('*--- POST PROCESSING ---*')
print('*-----------------------*')
#
#--- define general plot settings
#
import matplotlib as mpl
mpl.rcParams['font.size'] = 14.0
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['lines.color'] = 'r'
mpl.rcParams['axes.grid'] = True
#
#--- plot loading curve (prescribed disp vs. reaction)
#
print('Loading curve...')
# create a new figure and associated axes
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
# plot internal displacement vs. pressure
reactions = np.array(reactions) #shape (nSteps x nNodesOnInteriorBoundary)
displacements = np.array(displacements) #shape (nSteps x nNodesOnInteriorBoundary)
internal_displacement = np.mean(displacements,axis=1) # all nodal disp. are equal, so the mean = rdisp
internal_pressure = reactions.sum(axis=1)/(np.pi*Ri/2)
# compute theoretical result pi = c1*u(Ri) + c2 (cf. MEMCO TD9 or TP3)
lam = material.getLame1() ; mu = material.getLame2()
c0 = mu*Ri**3 + (mu+lam)*Re**2*Ri #constant
c1 = -2*mu*(lam+mu)*(Ri-Re)*(Ri+Re)/c0 #constant
internal_pressure_th = c1*internal_displacement
# plot
ax1.plot(internal_displacement*1e3,internal_pressure_th*1e-6,'-k',label="LinElast theory") # plot in (mm) and (MPa)
ax1.plot(internal_displacement*1e3,internal_pressure*1e-6,'--o',label="FE") # plot in (mm) and (MPa)
#configure plot
ax1.set_xlabel('Prescribed displacement $u_d$ (mm)')
ax1.set_ylabel('Internal pressure (MPa)')
ax1.legend() #add a legend
# savefig to .png file, at the same spot as the .msh file
fig_out1 = outputfile[:-4]+'_loading-curve.png'
fig1.tight_layout()
fig1.savefig(fig_out1,dpi=300,bbox_inches='tight')
fig1.show()
#
#--- plot evolution of the displacement along the bottom line
#
print('Displacement...')
# get nodes number and first coordinates along the bottom line
Ymin = np.min([mesh.getNode(i).getX(1) for i in range(nNodes)])
bottomLine = np.array([(i,mesh.getNode(i).getX(0)) for i in range(nNodes) if mesh.getNode(i).getX(1)<=Ymin])
bottomNodes = bottomLine[:,0].astype('int')
bottomCoordX = bottomLine[:,1]
# get nodal displacement along the bottom line
bottomU = U.reshape(nNodes,dim)[bottomNodes][:,0].flatten()
# compute theoretical result u(r) = a*r + b/r (cf. MEMCO TD9 or TP3)
lam = material.getLame1() ; mu = material.getLame2()
pi = internal_pressure[-1] #interior pressure at last time step
Req = 1/(Re**2-Ri**2) ; A = Ri**2*pi*Req ; B = Ri**2*Re**2*Req*pi #constants
a = A/(2*(lam+mu)) ; b = B/(2*mu) # constants
bottomU_linTH = a * bottomCoordX + b / bottomCoordX # theoretical nodal displacements u_r = a*r + b/r
# create figure and axis
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
# sort values according to ascending X-coordinates for the plot
ind = np.unravel_index(np.argsort(bottomCoordX,axis=None),bottomCoordX.shape)
# plot
ax2.plot(bottomCoordX[ind]*1e3,bottomU_linTH[ind]*1e3,'-k',label="LinElast theory")
ax2.plot(bottomCoordX[ind]*1e3,bottomU[ind]*1e3,'--o',label="FE")
#configure plot 
ax2.set_xlabel('Coordinate along $X$-axis (mm)')
ax2.set_ylabel('Prescribed displacement $u_d$ (mm)')
ax2.legend() #add a legend
# savefig to .png file, at the same spot as the .msh file
fig_out2 = outputfile[:-4]+'_disp_compare_th.png'
fig2.tight_layout()
fig2.savefig(fig_out2,dpi=300,bbox_inches='tight')
fig2.show()
