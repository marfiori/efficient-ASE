#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:12:01 2022

@author: flarroca
"""

import numpy as np
import scipy
from graspologic.embed import AdjacencySpectralEmbed 

# import ase_opt
import spectral_embedding_methods as sem

class out_of_sample_extension_np():
    """
    An class that in initialized with an adjacency matrix, and then it has several methods 
    to update the corresponding embedding(s)
    """
    
    def __init__(self, graph, initial_embedding_method='gp', M=None, d=None, max_iter=1000, tol=1e-6, b = 0.3, sigma = 0.1, t = 0.1):
        """
        We provide an initial graph, and then a first embedding is estimated

        Parameters
        ----------
        graph : numpy array of shape (n,n)
            The initial adjacency matrix.
        initial_embedding_method : str
            The method to use to calculate the initial embedding. It may be
            gp to use Graspologic or gd to use Gradient Descent. It will check
            whether the graph is directed or not.
        M : np.array with the same size as the graph
            The entries that should be considered when computing the loss for 
            the embedding. Please note that, for the initial embedding, it will 
            only considered de if initial_embedding_method is 'gp'.
        d : int
            The dimension of the embeddings. If None, if will be estimated.
        max_iter : int
            Maximum number of iterations for the gradient-descent embedding computation.
        tol : float
            Stopping criterion for the gradient-descent embedding computation.
        b : float
            beta parameter for the Armijo stepsize rule.
        sigma : float
            sigma parameter for the Armijo stepsize rule.
        t : float
            initial stepsize for the Armijo rule.


        Returns
        -------
        None.

        """
        
        self.n = graph.shape[0]
        if M is None:
            self.M = np.ones(self.n) - np.eye(self.n)
        else: 
            self.M = M
            
        self.max_iter = max_iter
        self.tol = tol
        self.b = b
        self.sigma = sigma
        self.t = t
        
        self.adj_matrix = graph
        
        if initial_embedding_method == 'gp':
            if d is None:
                ase = AdjacencySpectralEmbed(n_elbows=2, diag_aug=True, algorithm='full')
            else: 
                ase = AdjacencySpectralEmbed(n_components=d, diag_aug=True, algorithm='full')
            Xhats = ase.fit_transform(self.adj_matrix)
            if type(Xhats) is tuple:
                self.directed = True
                (self.Xlhat, self.Xrhat) = Xhats
                self.d = self.Xlhat.shape[1]
            else: 
                self.directed = False
                self.Xhat = Xhats
                self.d = self.Xhat.shape[1]
                
                # I compute the matrix Q
                # w = self.ase.singular_values_
                (w,v) = scipy.sparse.linalg.eigs(self.adj_matrix, k=self.d,which='LM')
                wabs = np.array(list(zip(-np.abs(w), -np.sign(np.real(w)))), dtype=[('abs', 'f4'), ('sign', 'i4')])
                w = w[np.argsort(wabs,order=['abs','sign'])]
                self.Q = np.diag(np.sign(w.real))
                # self.eigs = np.real(w)

        else:
            if not np.allclose(self.adj_matrix,self.adj_matrix.T):
                self.directed = True
                if d is None:
                    self.d = self.estimate_d()
                else: 
                    self.d = d
                X0 = np.random.rand(self.n, self.d)     
                # TODO convert it to iterative choice of both d and the embeddings and evaluate other random initializations
                self.Xlhat, self.Xrhat = sem.orthogonal_gradient_descent_DRDPG(self.adj_matrix, X0, X0, self.M, max_iter=self.max_iter,
                                                                               tol=self.tol, b=self.b, sigma=self.sigma, t=self.t)
                
            else: 
                self.directed = False
                if d is None:
                    self.d = self.estimate_d()
                else: 
                    self.d = d

                X0 = np.random.rand(self.n, self.d)
                # X0 = np.random.normal(size=(self.n, self.d))
                self.Q = np.array([1])
                for dim in np.arange(self.d):
                    # print(X0)
                    Q1 = np.eye(dim+1)
                    Q1[0:dim,0:dim] = self.Q
                    # Xhat_1 = sem.coordinate_descent_GRDPG(A=self.adj_matrix, X=np.random.rand(self.n, dim+1),  d=dim+1, Q=Q1, M=self.M)
                    Xhat_1 = sem.coordinate_descent_GRDPG(A=self.adj_matrix, X=X0[:,0:dim+1],  d=dim+1, Q=Q1, M=self.M, max_iter=self.max_iter,
                                                          tol=self.tol, b=self.b, sigma=self.sigma, t=self.t)
                    
                    Qminus1 = -np.eye(dim+1)
                    Qminus1[0:dim,0:dim] = self.Q
                    # Xhat_minus1 = sem.coordinate_descent_GRDPG(A=self.adj_matrix, X=np.random.rand(self.n, dim+1), d=dim+1, Q=Qminus1, M=self.M)
                    Xhat_minus1 = sem.coordinate_descent_GRDPG(A=self.adj_matrix, X=X0[:,0:dim+1], d=dim+1, Q=Qminus1, M=self.M, max_iter=self.max_iter,
                                                               tol=self.tol, b=self.b, sigma=self.sigma, t=self.t)
                    
                    if sem.cost_function_GRDPG(self.adj_matrix,Xhat_1,Q1,self.M) > sem.cost_function_GRDPG(self.adj_matrix,Xhat_minus1,Qminus1,self.M):
                        self.Q = Qminus1
                        self.Xhat = Xhat_minus1
                    else:
                        self.Q = Q1
                        self.Xhat = Xhat_1
                    print(self.Q)
                    
                    X0[:,0:dim+1] = self.Xhat



    def estimate_d(self):
        """
        Estimate the dimension for the embedding of the current graph.

        Returns
        -------
        int
            The estimated d.

        """
        # TODO Do it without computing the spectral decomposition
        ase = AdjacencySpectralEmbed(n_elbows=2, diag_aug=True, algorithm='full')
        Xhats = ase.fit_transform(self.adj_matrix)
        if self.directed:
            return Xhats[0].shape[1]
        else:
            return Xhats.shape[1]
        
    def add_new_nodes(self, hor_entries, ver_entries, hor_M=None, ver_M=None):
        """
        It adds new nodes to the graph, which are then embedded using either 
        gradient/coordinate descent (if the graph is undirected) or the
        manifold-esque optimization if directed. 

        Parameters
        ----------
        hor_entries : numpy array of shape (new_nodes,new_nodes+old_nodes)
            The new entries in the adjacency matrix that go below the old one.
        ver_entries : numpy array of shape (new_nodes+old_nodes,new_nodes)
            The new entries in the adjacency matrix that go to the right of the old one.
            Note that hor_entries and ver_entries are actually redundant, since they include the
            same values at the end (the interconnection between the new nodes). This is 
            not verified and it's up to you to check it.
        hor_M : numpy array of shape (new_nodes,new_nodes+old_nodes), optional
            The new entries in the mask matrix that go below the old one. The default is None.
        ver_M : numpy array of shape (new_nodes,new_nodes+old_nodes), optional
            The new entries in the mask that go to the right of the old one. The default is None.
            See the comment on ver_entries regarding the redundancy of this definition which applies here.

        Raises
        ------
        Exception
            If hor_entries does not have the same shape as ver_entries.T. If hor_M and hor_V don't have
            the same shape as hor_entries and ver_entries respectively. If the original graph was undirected
            then hor_entries and ver_entries should be such that hor_entries==ver_entries.T.

        Returns
        -------
        None.

        """
        if not self.directed:
            if not np.array_equal(hor_entries, ver_entries.T):
                raise Exception("The original graph was undirected. You should respect that!")
        if not np.array_equal(hor_entries.shape, ver_entries.T.shape):
            raise Exception("New entries should have a shape (n_new,n+n_new) and \
                            (n+n_new,n), even if the original graph was undirected.")
        if ver_M is not None:
            if (not np.array_equal(hor_entries.shape, hor_M.shape)) or (not np.array(ver_entries.shape, ver_M.shape)):
                raise Exception("The shape of the masks hor_M and ver_M should be the same as the \
                                shape of the new entries.")
        
        ## I first update the adjacency matrix with the new entries
        new_nodes = hor_entries.shape[0]
        new_adj_matrix = np.zeros((self.n+new_nodes,self.n+new_nodes))
        new_adj_matrix[0:self.n,0:self.n] = self.adj_matrix
        
        print("new nodes: "+str(new_nodes))
        
        new_adj_matrix[self.n:,:] = hor_entries
        new_adj_matrix[:,self.n:] = ver_entries
        
        self.adj_matrix = new_adj_matrix
        
        ## I then update the mask
        if hor_M is None:
            hor_M = np.concatenate((np.ones((new_nodes,self.n)),np.ones((new_nodes,new_nodes))-np.eye(new_nodes)), axis=1)
            ver_M = hor_M.T
        
        new_M = np.zeros((self.n+new_nodes,self.n+new_nodes))
        new_M[0:self.n,0:self.n] = self.M

        new_M[self.n:,:] = hor_M
        new_M[:,self.n:] = ver_M
        
        self.M = new_M
        
        self.n = self.n + new_nodes
        
        ### I now compute the new embeddings
        if self.directed:
            
            # I initialize the embeddings by projecting the new rows into the 
            # already estimated (as done by graspologic)
            proj_matrix_l = self.Xlhat/(np.linalg.norm(self.Xlhat,axis=0)**2)
            proj_matrix_r = self.Xrhat/(np.linalg.norm(self.Xrhat,axis=0)**2)
            new_Xl = np.concatenate((self.Xlhat,hor_entries[:,0:self.n-new_nodes]@proj_matrix_l))
            new_Xr = np.concatenate((self.Xrhat,ver_entries[0:self.n-new_nodes,:].T@proj_matrix_r))
            
            # # I initialize the embeddings randomly
            # new_Xl = np.concatenate((self.Xlhat,np.random.rand(new_nodes, self.d)))
            # new_Xr = np.concatenate((self.Xrhat,np.random.rand(new_nodes, self.d)))
            
            self.Xlhat, self.Xrhat = sem.orthogonal_gradient_descent_DRDPG(self.adj_matrix,new_Xl,new_Xr,new_M, max_iter=self.max_iter,
                                                                           tol=self.tol, b=self.b, sigma=self.sigma, t=self.t)
        else:
            # I initialize the embeddings by projecting the new rows into the 
            # already estimated (as done by graspologic)
            # TODO ver columnas ortogonales usando variedades. vale la pena??
            proj_matrix = self.Xhat/(np.linalg.norm(self.Xhat,axis=0)**2)
            
            new_X = np.concatenate((self.Xhat,hor_entries[:,0:self.n-new_nodes]@proj_matrix))
            # new_X = np.concatenate((self.Xhat,np.random.rand(new_nodes, self.d)))
            self.Xhat = sem.gradient_descent_GRDPG(self.adj_matrix,new_X,self.Q,self.M, max_iter=self.max_iter,
                                                   tol=self.tol, b=self.b, sigma=self.sigma, t=self.t)

    def add_new_nodes_and_lstsq_embed(self, hor_entries, ver_entries, hor_M=None, ver_M=None):
        """
        It adds new nodes to the graph, which are then embedded by a simple projection

        Parameters
        ----------
        hor_entries : numpy array of shape (new_nodes,new_nodes+old_nodes)
            The new entries in the adjacency matrix that go below the old one.
        ver_entries : numpy array of shape (new_nodes+old_nodes,new_nodes)
            The new entries in the adjacency matrix that go to the right of the old one.
            Note that hor_entries and ver_entries are actually redundant, since they include the
            same values at the end (the interconnection between the new nodes). This is 
            not verified and it's up to you to check it.
        hor_M : numpy array of shape (new_nodes,new_nodes+old_nodes), optional
            The new entries in the mask matrix that go below the old one. The default is None.
        ver_M : numpy array of shape (new_nodes,new_nodes+old_nodes), optional
            The new entries in the mask that go to the right of the old one. The default is None.
            See the comment on ver_entries regarding the redundancy of this definition which applies here.

        Raises
        ------
        Exception
            If hor_entries does not have the same shape as ver_entries.T. If hor_M and hor_V don't have
            the same shape as hor_entries and ver_entries respectively. If the original graph was undirected
            then hor_entries and ver_entries should be such that hor_entries==ver_entries.T.

        Returns
        -------
        None.

        """
        if not self.directed:
            if not np.array_equal(hor_entries, ver_entries.T):
                raise Exception("The original graph was undirected. You should respect that!")
        if not np.array_equal(hor_entries.shape, ver_entries.T.shape):
            raise Exception("New entries should have a shape (n_new,n+n_new) and \
                            (n+n_new,n), even if the original graph was undirected.")
        if ver_M is not None:
            if (not np.array_equal(hor_entries.shape, hor_M.shape)) or (not np.array_equal(ver_entries.shape, ver_M.shape)):
                raise Exception("The shape of the masks hor_M and ver_M should be the same as the \
                                shape of the new entries.")
        
        ## I first update the adjacency matrix with the new entries
        new_nodes = hor_entries.shape[0]
        new_adj_matrix = np.zeros((self.n+new_nodes,self.n+new_nodes))
        new_adj_matrix[0:self.n,0:self.n] = self.adj_matrix
                
        new_adj_matrix[self.n:,:] = hor_entries
        new_adj_matrix[:,self.n:] = ver_entries
        
        self.adj_matrix = new_adj_matrix
        
        ## I then update the mask
        if hor_M is None:
            hor_M = np.concatenate((np.ones((new_nodes,self.n)),np.ones((new_nodes,new_nodes))-np.eye(new_nodes)), axis=1)
            ver_M = hor_M.T
        
        new_M = np.zeros((self.n+new_nodes,self.n+new_nodes))
        new_M[0:self.n,0:self.n] = self.M

        new_M[self.n:,:] = hor_M
        new_M[:,self.n:] = ver_M
        
        self.M = new_M
        
        self.n = self.n + new_nodes
        
        ### I now compute the new embeddings
        if self.directed:
            
            # I initialize the embeddings by projecting the new rows into the 
            # already estimated (as done by graspologic)
            proj_matrix_l = self.Xlhat/(np.linalg.norm(self.Xlhat,axis=0)**2)
            proj_matrix_r = self.Xrhat/(np.linalg.norm(self.Xrhat,axis=0)**2)
            new_Xl = np.concatenate((self.Xlhat,hor_entries[:,0:self.n-new_nodes]@proj_matrix_r))
            new_Xr = np.concatenate((self.Xrhat,ver_entries[0:self.n-new_nodes,:].T@proj_matrix_l))
            
            self.Xlhat, self.Xrhat = new_Xl, new_Xr
        else:
            # I initialize the embeddings by projecting the new rows into the 
            # already estimated (as done by graspologic)
            # TODO ver columnas ortogonales usando variedades. vale la pena??
            proj_matrix = self.Xhat/(np.linalg.norm(self.Xhat,axis=0)**2)
            # print("xhat: ",self.Xhat)
            # print("(np.linalg.norm(self.Xhat,axis=0)**2): ",(np.linalg.norm(self.Xhat,axis=0)**2))
            # print("proj_matrix: ",proj_matrix)
            new_X = np.concatenate((self.Xhat,hor_entries[:,0:self.n-new_nodes]@proj_matrix))
            
            self.Xhat = new_X
            
    def add_one_node_and_lstsq_embed(self, new_row, new_col=None, new_m_row=None, new_m_col=None):
        """
        This method adds one node at a time and embeds them using the least-square method 
        as described in Eq. 4 of Keith Levin, Fred Roosta, Michael Mahoney, Carey Priebe, 
        "Out-of-sample extension of graph adjacency spectral embedding" 
        Proceedings of the 35th International Conference on Machine Learning, PMLR 80:2975-2984, 2018.
        
        Some adaptations were necessary to consider the generalized and directed case. 

        Parameters
        ----------
        new_row : numpy array of shape (1,n), where n is the current number of nodes.
            The new row to be added to the adjacency matrix (except for the 0 at the end).
        new_col : numpy array of shape (n,1), where n is the current number of nodes, optional
            The new column to be added to the adjacency matrix. The default is None, since it is
            not used when the graph is undirected.
        new_m_row : numpy array of shape (1,n).
            The row to be added to the mask (not used on the optimization, but updates this object's mask)
        new_m_col : numpy array of shape (n,1).
            The column to be added to the mask (not used on the optimization, but updates this object's mask)

        Returns
        -------
        None.

        """
        
        if not self.directed:
            if new_col is not None:
                raise Exception("You should not include a new row when the graph is undirected.")
            if new_m_col is not None:
                raise Exception("You should not include a new row when the graph is undirected.")
            # We have to solve argmin_{w\in \reals^d} \sum_{i=1}^n(a_i - X_i^t Q w)^2
            newX, _, _, _ = np.linalg.lstsq(self.Xhat@self.Q,new_row.T,rcond=None)
                        
            # I add the new embedding to this object's embeddings
            self.Xhat = np.concatenate((self.Xhat,newX.T))
            # and I update the adjacency matrix
            self.n = self.n+1
            new_adj_matrix = np.zeros((self.n,self.n))
            new_adj_matrix[0:self.n-1, 0:self.n-1] = self.adj_matrix
            new_adj_matrix[-1,0:-1] = new_row
            new_adj_matrix[0:-1,-1] = new_row
            self.adj_matrix = new_adj_matrix
            
            # and finally I update the mask
            if new_m_row is None:
                new_m_row = np.ones((1,self.n))
                new_m_row[0,-1] = 0
            new_M = np.ones((self.n,self.n))
            new_M[0:self.n-1,0:self.n-1] = self.M
            new_M[-1,:] = new_m_row
            new_M[:,-1] = new_m_row
            self.M = new_M
        
        else:
            if new_col is None:
                raise Exception("You must specify both the new row and column when using directed graphs")
                
            # we must solve argmin{wl,wr \in \reals^d} \sum_{i=1}^n (row_i-wl(Xr)_i^t)^2 + (col_i-wr(Xl)_i^t)^2
            # To do this, I solve a LS problem with a 2*d dimension vector, and place the coeffs and ordinate 
            # variables accordingly. 
            coeff = np.zeros((2*self.n,2*self.d))
            coeff[0:self.n,0:self.d] = self.Xlhat
            coeff[self.n:2*self.n,self.d:2*self.d] = self.Xrhat
            
            ordinate = np.concatenate((new_col, new_row.T))
            
            solution, _, _, _ = np.linalg.lstsq(coeff, ordinate, rcond=None)
            
            # Now I add the embeddings to this object's embeddings. 
            new_Xrhat = solution[0:self.d]
            new_Xlhat = solution[self.d:]
            
            # I add the new embeddings to this object's embeddings
            self.Xlhat = np.concatenate((self.Xlhat,new_Xlhat.T))
            self.Xrhat = np.concatenate((self.Xrhat,new_Xrhat.T))
            # and I update the adjacency matrix
            self.n = self.n+1
            new_adj_matrix = np.zeros((self.n,self.n))
            new_adj_matrix[0:self.n-1, 0:self.n-1] = self.adj_matrix
            new_adj_matrix[-1,0:-1] = new_row
            new_adj_matrix[0:-1,-1] = new_col.flatten()
            self.adj_matrix = new_adj_matrix
            
            # and finally I update the mask
            if new_m_row is None:
                new_m_row = np.ones((1,self.n))
                new_m_row[0,-1] = 0
                new_m_col = new_m_row.T
            new_M = np.ones((self.n,self.n))
            new_M[0:self.n-1,0:self.n-1] = self.M
            new_M[-1,:] = new_m_row
            new_M[:,-1] = new_m_col.flatten()
            self.M = new_M
            
            
    def update_adj_matrix_trip_basic(self,delta_adj):
        """
        This method implements Trip-Basic (First Order Eigen-Pairs Tracking)
        from Chen Chen and Hanghang Tong, "Fast Eigen-Functions Tracking on Dynamic Graphs," in
        2015 SIAM International Conference on Data Mining (SDM).
        
        Given a differential adjacency matrix it updates the ASE by estimating the variation 
        of the eigenvectors and eigenvalues. It should only be used if these are actually tracked 
        (i.e. it's not guaranteed to work if we used gradient descent methods). Its non-basic counterpart 
         should provide better results, specially if the eigengap is small. 

        Parameters
        ----------
        delta_adj : a numpy array of shape (n,n)
            The difference between the previous and the current adjacency matrix.

        Raises
        ------
        Exception
            If the original graph was undirected this method should not be used.

        Returns
        -------
        None.

        """
        if self.directed:
            raise Exception("TRIP-Basic updates eigen-values, so it only works for undirected graphs")            
        
        if np.linalg.norm(delta_adj,ord='fro')==0:
            return
        ## Warning: This will only work if I've kept track of the actual eigen-vectors!!
        eig_vectors = self.Xhat
        eig_vectors_next = eig_vectors.copy()
        eig_values = np.diag(eig_vectors.T@self.adj_matrix@eig_vectors/(eig_vectors.T@eig_vectors)).copy()
        eig_values_next = eig_values.copy()
        for j in np.arange(self.d):
            delta_u_j = np.zeros((self.n,1))
            for i in np.arange(self.d):
                if i != j:
                    delta_u_j += (eig_vectors[:,[i]].T@delta_adj@eig_vectors[:,[j]]/(eig_values[j]-eig_values[i]))*eig_vectors[:,[i]]
                    
            delta_lambda_j = eig_vectors[:,[i]].T@delta_adj@eig_vectors[:,[j]]
            
            eig_values_next[j] = eig_values[j] + delta_lambda_j 
            eig_vectors_next[:,[j]] += delta_u_j
            eig_vectors_next[:,[j]] = eig_vectors[:,[j]]/np.linalg.norm(eig_vectors[:,[j]])
            
        self.adj_matrix += delta_adj
        self.Xhat = eig_vectors_next@np.diag(np.sqrt(np.abs(eig_values_next)))
        
    def update_adj_matrix_trip(self, delta_adj):
        """
        This method implements Trip (High Order Eigen-Pairs Tracking)
        from Chen Chen and Hanghang Tong, "Fast Eigen-Functions Tracking on Dynamic Graphs," in
        2015 SIAM International Conference on Data Mining (SDM).
        
        Given a differential adjacency matrix it updates the ASE by estimating the variation 
        of the eigenvectors and eigenvalues. It should only be used if these are actually tracked 
        (i.e. it's not guaranteed to work if we used gradient descent methods). It should provide 
         better results than its basic counterpart, specially if the eigengap is small. 

        Parameters
        ----------
        delta_adj : a numpy array of shape (n,n)
            The difference between the previous and the current adjacency matrix.

        Raises
        ------
        Exception
            If the original graph was undirected this method should not be used.

        Returns
        -------
        None.

        """
        
        if self.directed:
            raise Exception("TRIP-Basic updates eigen-values, so it only works for undirected graphs")            
            
        if np.linalg.norm(delta_adj,ord='fro')==0:
            return
        ## Warning: This will only work if I've kept track of the actual eigen-vectors!!
        eig_vectors = self.Xhat
        eig_vectors_next = eig_vectors.copy()
        
        eig_values = np.diag(eig_vectors.T@self.adj_matrix@eig_vectors/(eig_vectors.T@eig_vectors)).copy()
        
        X = eig_vectors.T@delta_adj@eig_vectors
        delta_lam = np.diag(X).copy()
        eig_values_next = eig_values + delta_lam
        
        for j in np.arange(self.d):
            v = eig_values[j] + delta_lam[j] - eig_values
            D = np.diag(v)
            
            # print("D: ", D)
            # print("X: ", X)
            
            alpha_j = np.linalg.inv(D-X)@X[:,[j]]
            delta_u_j = eig_vectors@alpha_j
            eig_vectors_next[:,[j]] += delta_u_j
            eig_vectors_next[:,[j]] = eig_vectors[:,[j]]/np.linalg.norm(eig_vectors[:,[j]])
        
        self.adj_matrix += delta_adj
        self.Xhat = eig_vectors_next@np.diag(np.sqrt(np.abs(eig_values_next)))
        
        
    def update_adj_matrix_lwi_svd(self, delta_adj):
        """
        This method implements Low-Rank Windowed Incremental SVD from
        Xilun Chen and K. Selcuk Candan. "LWI-SVD: low-rank, windowed, incremental singular 
        value decompositions on time-evolving data sets". In ACM SIGKDD (KDD '14).
        
        It's basically the method described in 
        Matthew Brand, "Fast low-rank modifications of the thin singular value decomposition," 
        Linear Algebra and its Applications, 2006.
        
        where the delta_adj is written as the product of two matrices (delta_adj = AB^T), and 
        it considers the special case of A=I and B^T=delta_adj.

        Parameters
        ----------
        delta_adj : a numpy array of shape (n,n)
            The difference between the previous and the current adjacency matrix.

        Returns
        -------
        None.

        """
        
        if np.linalg.norm(delta_adj,ord='fro')==0:
            return
        if delta_adj.shape != (self.n,self.n):
            raise Exception("The array indicating the change should have the same size as the original matrix")
        
        if self.directed:
            U = self.Xlhat/np.linalg.norm(self.Xlhat,axis=0)
            V = self.Xrhat/np.linalg.norm(self.Xrhat,axis=0)
            sing_values = np.linalg.norm(self.Xlhat,axis=0)*np.linalg.norm(self.Xrhat,axis=0)
        else:
            if not np.array_equal(delta_adj, delta_adj.T):
                raise Exception("The original graph was undirected. You should respect that!")
                
            U = self.Xhat/np.linalg.norm(self.Xhat,axis=0)
            sing_values = np.linalg.norm(self.Xhat,axis=0)**2
            V = U
        # TODO the abs is necessary?
        S = np.diag(sing_values)
        
        P = scipy.linalg.orth((np.eye(self.n)-U@U.T))
        RA = P.T@((np.eye(self.n)-U@U.T))
        
        # Q = scipy.linalg.orth((np.eye(self.n)-V@V.T)@self.adj_matrix.T)
        # RB = Q.T@((np.eye(self.n)-V@V.T)@self.adj_matrix.T)
        Q = scipy.linalg.orth((np.eye(self.n)-V@V.T)@delta_adj.T)
        RB = Q.T@((np.eye(self.n)-V@V.T)@delta_adj.T)
        
        K = np.vstack((U.T,RA))@(np.vstack((V.T@delta_adj.T,RB)).T)
        pad = np.array(K.shape)-np.array(S.shape)
        K = np.pad(S,((0,pad[0]), (0,pad[1]))) + K
        
        # (UK, SK, VKt) = np.linalg.svd(K)
        # Use instead scipy's svds method that computes only the k largest SVs
        (UK, SK, VKt) = scipy.sparse.linalg.svds(K,k=self.d, which='LM')
        # Order is not guaranteed
        index = np.flip(np.argsort(SK))
        UK = UK[:,index]
        VKt = VKt[index,:]
        SK = SK[index]
        # UK = UK[:,0:self.d]
        # VKt = VKt[0:self.d,:]
        # SK = SK[0:self.d]
        
        new_U = np.hstack((U,P))@UK
        new_S = np.diag(SK)
            
        if self.directed:
            new_V = np.hstack((V,Q))@(VKt.T)
            self.Xlhat = np.reshape(new_U[0:self.n,0:self.d]@np.sqrt(new_S[0:self.d,0:self.d]),(self.n,self.d))
            self.Xrhat = np.reshape(new_V[0:self.n,0:self.d]@np.sqrt(new_S[0:self.d,0:self.d]),(self.n,self.d))
        else:            
            self.Xhat = np.reshape(new_U[0:self.n,0:self.d]@np.sqrt(new_S[0:self.d,0:self.d]),(self.n,self.d))
        
        self.adj_matrix += delta_adj
                
        

    def update_adj_matrix(self, delta_adj, delta_M = None):
        """
        This method updates the adjacency matrix and mask (e.g. self.adj_matrix += delta_adj)
        and then runs a GD to update the embeddings (hot-started from the previous estimation). 

        Parameters
        ----------
        delta_adj : numpy array of shape (n,n)
            The difference between current and new adjacency matrix.
        delta_M : numpy array of shape (n,n), optional
            The difference between current and new mask. The default, which amounts 
            to no change, is None.

        Raises
        ------
        Exception
            If shapes don't match, or if you're trying to use a non-symmetric
            change on an undirected object.

        Returns
        -------
        None.

        """
        if not self.directed:
            if not np.array_equal(delta_adj, delta_adj.T):
                raise Exception("The original graph was undirected. You should respect that!")
                
        if not np.array_equal(self.adj_matrix.shape,delta_adj.shape):
            raise Exception("delta_adj should be the same size as the original adjacency matrix.")
        
        if delta_M is not None:
            if not np.array_equal(delta_adj.shape, delta_M.shape):
                raise Exception("The shape of the mask should be the same as the \
                                shape of the new entries.")
        
        ## I first update the adjacency matrix with the new entries
        
        self.adj_matrix += delta_adj
        
        ## I then update the mask
        if delta_M is not None:
            self.M += delta_M
        
        ### I now compute the new embeddings
        if self.directed:
            self.Xlhat, self.Xrhat = sem.orthogonal_gradient_descent_DRDPG(self.adj_matrix,self.Xlhat,self.Xrhat,self.M,
                                                                           max_iter=self.max_iter, tol=self.tol, b=self.b,
                                                                           sigma=self.sigma, t=self.t)
        else:
            self.Xhat = sem.gradient_descent_GRDPG(self.adj_matrix,self.Xhat,self.Q,self.M, max_iter=self.max_iter,
                                                   tol=self.tol, b=self.b, sigma=self.sigma, t=self.t)       
        
    def remove_nodes(self, nodes, update=False):
        """
        This method updates removes some nodes and optionally updates the 
        embeddings

        Parameters
        ----------
        nodes : list
            The indices of nodes to remove
        update : bool, optional
            If the embeddings should be updated or not

        """
        
        # I update the adjacency matrix
        self.adj_matrix = np.delete(self.adj_matrix,nodes, axis=0)
        self.adj_matrix = np.delete(self.adj_matrix,nodes, axis=1)
        
        # I update the mask
        self.M = np.delete(self.M, nodes, axis=0)
        self.M = np.delete(self.M, nodes, axis=1)
        
        # I remove the embeddings
        if self.directed:
            self.Xlhat = np.delete(self.Xlhat, nodes, axis=0)
            self.Xrhat = np.delete(self.Xrhat, nodes, axis=0)
        else:
            self.Xhat = np.delete(self.Xhat, nodes, axis=0) 
        
        self.n -= len(nodes)
        
        if update:
            ### I now update the new embeddings
            if self.directed:
                self.Xlhat, self.Xrhat = sem.orthogonal_gradient_descent_DRDPG(self.adj_matrix,self.Xlhat,self.Xrhat,self.M,
                                                                               max_iter=self.max_iter, tol=self.tol, b=self.b,
                                                                               sigma=self.sigma, t=self.t)
            else:
                self.Xhat = sem.gradient_descent_GRDPG(self.adj_matrix,self.Xhat,self.Q,self.M,
                                                       max_iter=self.max_iter, tol=self.tol, b=self.b,
                                                       sigma=self.sigma, t=self.t)  
                    