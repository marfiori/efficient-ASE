#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
import scipy
#Install the following fork of pymanopt in order to use the manifold of matrices with orthogonal columns
#pip install git+https://github.com/marfiori/pymanopt
from pymanopt.manifolds import Stiefel_tilde

def coordinate_descent(A,d,X=None,tol=1e-5):
    """
    Solves the problem  min ||(A-XX^T)*M||_F^2
    by block coordinate descent.
    Here * is the entry-wise product.
    M is the matrix with zeros in the diagonal, and ones off-diagonal.    
    Returns X, solution of min ||(A-XX^T)*M||_F^2
    
    Parameters
    ----------
    A : matrix nxn
    d : dimension of the embedding
    X : initialization
    tol: tolerance used in the stop criterion  
        
    Returns
    -------
    Matrix X
        solution of the embedding problem
    """

    n=A.shape[0]
    M = np.ones(n) - np.eye(n)
    if X is None:
        X = np.random.rand(n,d)
    else:
        X = X.copy()
    
    R = X.T@X
    fold = -1
    while (abs((fold - cost_function(A, X, M))/fold) >tol):
        fold = cost_function(A, X, M)
        for i in range(n):
            k=X[i,:][np.newaxis]
            R -= k.T@k
            X[i,:] = solve_linear_system(R,(A[i,:]@X).T,X[i,:])
            k=X[i,:][np.newaxis]
            R += k.T@k

    return X


def orthogonal_gradient_descent_RDPG(A,X,M,tol=1e-3):
    """
    Solves the problem  min ||(A-XX^T)*M||_F^2 with the constraint of X having orthogonal columns,
    by gradient descent on the Riemannian manifold of matrices with orthogonal columns.
    Here * is the entry-wise product.
    
    Parameters
    ----------
    A : matrix nxn
    X : initialization
    M : mask matrix nxn
    tol: tolerance used in the stop criterion  
        
    Returns
    -------
    Matrix X
        solution of the embedding problem
    """
    b=0.3; sigma=0.1 # Armijo parameters
    rank = X.shape[1]
    max_iter = 200*rank
    t = 0.1
    Xd=X
    k=0
    last_jump=1
    
    manifold = Stiefel_tilde(X.shape[0],X.shape[1])
    d = manifold.projection(Xd,-gradient(A,Xd,M))
    funct_vals = []
    grad_vals = []
    while (la.norm(d) > tol) & (last_jump > 1e-16) & (k<max_iter):

        # Armijo rule
        while (cost_function(A, manifold.retraction(Xd,t*d), M) > cost_function(A, Xd, M) - sigma*t*la.norm(d)**2):
            t=b*t

        Xd = manifold.retraction(Xd,t*d)
        last_jump = sigma*t*la.norm(d)**2
        t=t/(b)
        k=k+1
        d = manifold.projection(Xd,-gradient(A,Xd,M))
        funct_vals.append(cost_function(A, Xd, M))
        grad_vals.append(la.norm(d))
    return(Xd)


def gradient_descent_RDPG(A,X,M, tol=1e-3):
    """
    Solves the problem  min ||(A-XX^T)*M||_F^2 without any constraint on X,
    by classical gradient descent.
    Here * is the entry-wise product.
    
    Parameters
    ----------
    A : matrix nxn
    X : initialization
    M : mask matrix nxn
    tol: tolerance used in the stop criterion  
        
    Returns
    -------
    Matrix X
        solution of the embedding problem
    """
    b=0.3; sigma=0.1 # Armijo parameters
    rank = X.shape[1]
    max_iter = 200*rank
    t = 0.1
    Xd=X
    k=0
    last_jump=1
    d = -gradient(A,Xd,M)
    tol = tol*(la.norm(d))
    while (la.norm(d) > tol) & (last_jump > 1e-16) & (k<max_iter):

        # Armijo
        while (cost_function(A, Xd+t*d, M) > cost_function(A, Xd, M) - sigma*t*la.norm(d)**2):
            t=b*t

        Xd = Xd+t*d
        last_jump = sigma*t*la.norm(d)**2
        t=t/(b)
        k=k+1
        d = -gradient(A,Xd,M)
    return(Xd)


def gradient_descent_GRDPG(A, X, Q, M, max_iter=100, tol=1e-3, b=0.3, sigma=0.1, t=0.1):
    """
    Solves the problem min ||(A-XQX^T)*M||_F^2 by classical gradient descent.
    Here * is the entry-wise product.
    
    Parameters
    ----------
    A : matrix nxn
    X : initialization
    Q : diagonal matrix with elements +1 or -1
    M : mask matrix nxn
    max_iter: maximum number of iterations
    tol: tolerance used in the stop criterion
    b: beta parameter for the Armijo stepsize rule
    sigma: sigma parameter for the Armijo stepsize rule
    t: initial stepsize for the Armijo rule
        
    Returns
    -------
    Matrix X
        solution of the embedding problem
    """

    Xd=X
    k=0
    last_jump=1
    d = -gradient_GRDPG(A,Xd,Q,M)
    while (la.norm(d) > tol) & (last_jump > 1e-16) & (k<max_iter):

        # Armijo rule
        while (cost_function_GRDPG(A, Xd+t*d,Q, M) > cost_function_GRDPG(A, Xd,Q, M) - sigma*t*la.norm(d)**2):
            t=b*t

        Xd = Xd+t*d
        last_jump = sigma*t*la.norm(d)**2
        t=t/(b)
        k=k+1
        d = -gradient_GRDPG(A,Xd,Q,M)
    return(Xd)

def coordinate_descent_GRDPG(A,d,Q,M=None,X=None,max_iter=100,tol=1e-5):
    """
    Solves the problem  min ||(A-XQX^T)*M||_F^2
    by block coordinate descent.
    
    Returns X, solution of min ||(A-XQX^T)*M||_F^2
    
    Parameters
    ----------
    A : matrix nxn
    d : dimension of the embedding
    Q : diagonal matrix with values +1 or -1
    M : mask matrix
    X : initialization
    max_iter: maximum number of iterations
    tol: tolerance used in the stop criterion  
        
    Returns
    -------
    Matrix X
        solution of the embedding problem
    """
    
    n=A.shape[0]
    if X is None:
        X = np.random.rand(n,d)
    else:
        X = X.copy()
    if M is None:
        M = np.ones(n) - np.eye(n)
    fold = 1
    
    k = 0

    while (abs((fold - cost_function_GRDPG(A, X, Q, M))/fold) >tol) & (k<max_iter):
        fold = cost_function_GRDPG(A, X, Q, M)
        for i in range(n):
            k2 = Q@(np.broadcast_to([M[i,:]], (d, n))*X.T)
            X[i,:] = solve_linear_system(k2@k2.T,(M*A)[i,:]@X@Q,X[i,:])
        k=k+1
    return X

def orthogonal_gradient_descent_DRDPG(A, Xl, Xr, M, max_iter = 100, tol=1e-6, b = 0.3, sigma = 0.1, t = 0.1):
    """
    Solves the directed RDPGs embedding problem min ||(A - Xl Xr^T)*M||_F^2 with the constraint of Xl and Xr having orthogonal columns,
    by gradient descent on the Riemannian manifold of matrices with orthogonal columns.
    Here * is the entry-wise product.
    
    Parameters
    ----------
    A : matrix nxn
    Xl : initialization of left embeddings
    Xr : initialization of right embeddings
    M : mask matrix nxn
    max_iter: maximum number of iterations
    tol: tolerance used in the stop criterion  
    b: beta parameter for the Armijo stepsize rule
    sigma: sigma parameter for the Armijo stepsize rule
    t: initial stepsize for the Armijo rule
        
    Returns
    -------
    Matrices Xl and Xr
        solution of the embedding problem
    """
    k=0
    last_jump=1
    manifold = Stiefel_tilde(Xl.shape[0],Xl.shape[1])
    
    Gl = manifold.projection(Xl,-( (Xl@Xr.T-A)*M )@Xr)
    Gr = manifold.projection(Xr,-(( (Xl@Xr.T-A)*M ).T)@Xl)

    max_iter_cost = 100

    while (la.norm(Gl) + la.norm(Gr) > tol) & (last_jump > 1e-16) & (k<max_iter):

        Gl = manifold.projection(Xl,-( (Xl@Xr.T-A)*M )@Xr)
        # Armijo
        
        count = 0
        while (cost_function_DRDPG(A, manifold.retraction(Xl,t*Gl),Xr, M) > cost_function_DRDPG(A, Xl,Xr, M) - sigma*t*la.norm(Gl)**2) & (count<max_iter_cost):
            t=b*t
            count = count + 1

        Xl = manifold.retraction(Xl,t*Gl)
        last_jump = sigma*t*la.norm(Gl)**2
        t=t/(b)
        
        Gr = manifold.projection(Xr,-(( (Xl@Xr.T-A)*M ).T)@Xl)
        
        count = 0
        while (cost_function_DRDPG(A, Xl,manifold.retraction(Xr,t*Gr),M) > cost_function_DRDPG(A, Xl,Xr, M) - sigma*t*la.norm(Gr)**2) & (count < max_iter_cost):
            t=b*t
            count = count + 1
            
        Xr = manifold.retraction(Xr,t*Gr)
        last_jump = last_jump + sigma*t*la.norm(Gr)**2
        t=t/(b)
        k=k+1
        
    # I finally normalize both embeddings
    # TODO should this be done here?
    (Xl, Xr) = normalize_rdpg_directive(Xl,Xr)    
    return Xl, Xr

def gradient_descent_DRDPG(A,Xl,Xr,M, tol=1e-6):
    """
    Solves the directed RDPGs embedding problem min ||(A - Xl Xr^T)*M||_F^2 without any constraint on Xl and Xr,
    by classical gradient descent.
    Here * is the entry-wise product.
    
    Observe that since this method does not guarantee the orthogonality of the resulting matrices, the embeddings may not be interpretable.
    
    Parameters
    ----------
    A : matrix nxn
    Xl : initialization of left embeddings
    Xr : initialization of right embeddings
    M : mask matrix nxn
    tol: tolerance used in the stop criterion  
        
    Returns
    -------
    Matrices Xl and Xr
        solution of the embedding problem (without orthogonality constraints)
    """
    b=0.3; sigma=0.1 # Armijo parameters
    max_iter = 100
    t = 0.1
    k=0
    last_jump=1
    
    Gl = -( (Xl@Xr.T-A)*M )@Xr
    Gr = -(( (Xl@Xr.T-A)*M ).T)@Xl
    while (la.norm(Gl) + la.norm(Gr) > tol) & (last_jump > 1e-16) & (k<max_iter):
        Gl = -( (Xl@Xr.T-A)*M )@Xr
        
        # Armijo
        while (cost_function_DRDPG(A, Xl+t*Gl,Xr, M) > cost_function_DRDPG(A, Xl,Xr, M) - sigma*t*la.norm(Gl)**2):
            t=b*t

        Xl = Xl+t*Gl
        last_jump = sigma*t*la.norm(Gl)**2
        t=t/(b)
        
        Gr = -(( (Xl@Xr.T-A)*M ).T)@Xl
        while (cost_function_DRDPG(A, Xl,Xr+t*Gr,M) > cost_function_DRDPG(A, Xl,Xr, M) - sigma*t*la.norm(Gr)**2):
            t=b*t
        Xr = Xr+t*Gr
        last_jump = last_jump + sigma*t*la.norm(Gr)**2
        t=t/(b)
        k=k+1
        
    return Xl, Xr


"""One step of Gradient descent for RDPGs."""
def gradient_descent_RDPG_one_step(A,X,M,t,armijo=0):
    """
    Makes one step of gradient descent for the problem  min ||(A-XX^T)*M||_F^2.
    Here * is the entry-wise product.
    
    Parameters
    ----------
    A : matrix nxn
    X : initial embedding matrix
    M : mask matrix nxn
    t : step size
    armijo: if 1, the Armijo rule is used to compute the step size. If 0, the provided step-size t is used
        
    Returns
    -------
    Matrix X
        embedding matrix X after one step of gradient descent from provided initial matrix.
    """

    b=0.3
    sigma=0.1
    d = -gradient(A,X,M)
    if armijo:
        while (cost_function(A, X+t*d, M) > cost_function(A, X, M) - sigma*t*la.norm(d)**2):
            t=b*t
    return(X+t*d)



##### Auxiliary functions #####

def gradient(A,X,M):
    """
    Gradient of the function ||(A-XX^T)*M||_F^2
    where * is the entry-wise product.

    Parameters
    ----------
    A : matrix nxn
    X : matrix of embeddings
    M : mask matrix nxn
        
    Returns
    -------
    a matrix, gradient of the function ||(A-XX^T)*M||_F^2
    """
    return 2*( -((M.T+M)*A) + ((M.T+M)*(X@X.T) ))@X

def gradient_GRDPG(A,X,Q,M):
    """
    Gradient of the function ||(A-XQX^T)*M||_F^2
    where * is the entry-wise product.

    Parameters
    ----------
    A : matrix nxn
    X : matrix of embeddings
    Q : diagonal matrix with elements +1 or -1
    M : mask matrix nxn
        
    Returns
    -------
    a matrix, gradient of the function ||(A-XX^T)*M||_F^2
    """
    return (2*((X@Q@X.T)*M.T) - (A*M*M + A.T*M.T))@X@Q

def cost_function(A,X,M):
    """
    RDPG cost function ||(A-XX^T)*M||_F^2
    where * is the entry-wise product.

    Parameters
    ----------
    A : matrix nxn
    X : matrix of embeddings
    M : mask matrix nxn
        
    Returns
    -------
    value of ||(A-XX^T)*M||_F^2
    """
    return 0.5*np.linalg.norm((A - X@X.T)*M,ord='fro')**2

def cost_function_GRDPG(A,X,Q,M):
    """
    Generalized RDPG cost function ||(A-XQX^T)*M||_F^2
    where * is the entry-wise product.
    
    Parameters
    ----------
    A : matrix nxn
    X : matrix of embeddings
    Q : diagonal matrix with elements +1 or -1
    M : mask matrix nxn
        
    Returns
    -------
    value of ||(A-XQX^T)*M||_F^2
    """
    return 0.5*np.linalg.norm((A - X@Q@X.T)*M,ord='fro')**2

def cost_function_DRDPG(A, Xl,Xr, M):
    """
    Directed-RDPG cost function ||(A - Xl Xr^T)*M||_F^2
    where * is the entry-wise product.

    Parameters
    ----------
    A : matrix nxn
    Xl : matrix of left embeddings
    Xr : matrix of right embeddings
    M : mask matrix nxn
        
    Returns
    -------
    value of ||(A - Xl Xr^T)*M||_F^2
    """

    return 0.5*np.linalg.norm((A - Xl@Xr.T)*M,ord='fro')**2

def solve_linear_system(A,b,xx):
    """
    Linear system solver, used in several methods.
    Should you use another method for solving linear systems, just change this function.
    
    Returns the solution of Ax=b
    Parameters
    ----------
    A : matrix nxn
    b : vector 

    Returns
    -------
    vector x
        solution to Ax=b

    """
    try:
        result = scipy.linalg.solve(A,b)
    except:
        result = scipy.sparse.linalg.minres(A,b,xx)[0]    
    return result


def rsvd(A,r,q,p):
    """
    randomSVD: Implemenation of a random SVD method.
    Used to compute an approximation of the SVD.    

    Parameters
    ----------
    A : matrix to decompose
    r : number of singular values to keep
    q : power iteration parameter (q=1 or q=2 may be enough)
    p : oversampling factor


    Returns
    -------
    U, S, V^T, approximate SVD decomposition of A
    """

    ny = A.shape[1]
    P = np.random.randn(ny,r+p)
    Z = A @ P
    for k in range(q):
        Z = A @ (A.T @ Z)
        
    Q, R = la.qr(Z,mode='reduced')
    Y = Q.T @ A
    UY, S, VT = scipy.linalg.svd(Y)
    U = Q @ UY
    return U, S, VT

def align_Xs(X1,X2):
    """
    An auxiliary function that Procrustes-aligns two embeddings. 
    Parameters
    ----------
    X1 : an array-like with the embeddings to be aligned
    X2 : an array-like with the embeddings to align X1 to
    Returns
    -------
    X1_aligned : the aligned version of X1 to X2.
    """
    V,_,Wt = la.svd(X1.T@X2)
    U = V@Wt
    X1_aligned = X1@U
    return X1_aligned

def normalize_rdpg_directive(Xhatl,Xhatr):
    """
    An auxiliary function to normalize embeddings of directional graphs. 
    Parameters
    ----------
    Xhatl : an array-like with the left embeddings
    Xhatr : an array-like with the right embeddings
    Returns
    -------
    Xhatl : the normalized left embedding.
    Xhatr : the normalized right embeddings. 
    """
    dims = Xhatl.shape[1]
    for d in np.arange(dims):
        factor = np.sqrt(np.linalg.norm(Xhatl[:,d])/np.linalg.norm(Xhatr[:,d]))
        Xhatl[:,d] = Xhatl[:,d]/factor
        Xhatr[:,d] = Xhatr[:,d]*factor
    return (Xhatl, Xhatr)
