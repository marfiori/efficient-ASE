#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:12:01 2022

@author: flarroca
"""

import networkx as nx
import numpy as np
import logging as lg

from out_of_sample_extension.out_of_sample_extension_np import OutOfSampleExtensionNp

class OutOfSampleExtensionNx():
    """
    An class that in initialized with a graph, and then it has several methods 
    to update the corresponding embedding(s)
    """
    
    def __init__(self, graph, initial_embedding_method='gp', unknown_edge_attr=None, d=None, max_iter=1000, tol=1e-6, b = 0.3, sigma = 0.1, t = 0.1, verbose=True):
        """
        We provide an initial graph, and then a first embedding is estimated

        Parameters
        ----------
        graph : NetworkX graph
            The initial graph.
        initial_embedding_method : str
            The method to use to calculate the initial embedding. It may be
            gp to use Graspologic or gd to use Gradient Descent. It will check
            whether the graph is directed or not.
        unknown_edge_attr : str
            Edge attribute that holds whether each edge in graph is to be treated as an
            unknown (an thus should not be considered when computing the loss for 
            the embedding). Please note that, for the initial embedding, it will 
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
        
        # Save nodes's order for adjacency matrix creation and comparison
        self.nodes_in_order = list(graph.nodes())
        
        # Mask is obtained from the graph's edges atributes
        self.unknown_edge_attr = unknown_edge_attr
        M = None
        if self.unknown_edge_attr is not None:
            mask = nx.to_numpy_array(graph, nodelist=self.nodes_in_order, weight=self.unknown_edge_attr)
            M = np.ones_like(mask) - np.eye(mask.shape[0]) - mask
            
        self.ose = OutOfSampleExtensionNp(nx.to_numpy_array(graph,nodelist=self.nodes_in_order),
                                          initial_embedding_method=initial_embedding_method, M=M,
                                          d=d, max_iter=max_iter, tol=tol, b=b, sigma=sigma, t=t)
        self.verbose = verbose
        
        
    def get_embeddings(self):
        if self.ose.directed:
            return (self.ose.Xlhat, self.ose.Xrhat)
        else:
            return self.ose.Xhat
    
    def new_graph(self,new_G):
        
        nodes_new_G = new_G.nodes
        
        if set(nodes_new_G) == set(self.nodes_in_order):
            # nodes did not change
            delta_adj = nx.to_numpy_array(new_G,nodelist=self.nodes_in_order) - self.ose.adj_matrix
            
            # check if mask changed
            delta_M = None
            if self.unknown_edge_attr is not None:
                new_mask = nx.to_numpy_array(new_G,nodelist=self.nodes_in_order, weight=self.unknown_edge_attr)
                new_M  = np.ones_like(new_mask) - np.eye(new_mask.shape[0]) - new_mask
                delta_M =  new_M - self.ose.M
                
            self.ose.update_adj_matrix(delta_adj,delta_M)
            
        else:
            # I either lost or gained nodes (or both)

            
            ####### gained nodes #########
            new_nodes = list(set(nodes_new_G) - set(self.nodes_in_order))
            # the new nodes' list
            self.nodes_in_order += new_nodes
            # Original node count     
            original_n = self.ose.n
            
            if len(new_nodes)>0:
                # I gained new nodes
                if self.verbose:
                    print(f"From the original {original_n} nodes you've gained {len(new_nodes)} nodes.")
                
                # the removed nodes will appear as disconnected in adj_matrix_new
                adj_matrix_new = nx.to_numpy_array(new_G, nodelist=self.nodes_in_order)          
                new_rows = adj_matrix_new[np.arange(self.ose.n,self.ose.n+len(new_nodes),1),:]
                new_cols = adj_matrix_new[:,np.arange(self.ose.n,self.ose.n+len(new_nodes),1)]    
                
                # Update mask
                hor_M = None
                ver_M = None
                
                if self.unknown_edge_attr is not None:
                    new_mask = nx.to_numpy_array(new_G, nodelist=self.nodes_in_order, weight=self.unknown_edge_attr)
                    new_M  = np.ones_like(new_mask) - np.eye(new_mask.shape[0]) - new_mask
                    hor_M = new_M[np.arange(self.ose.n,self.ose.n+len(new_nodes),1),:]
                    ver_M = new_M[:,np.arange(self.ose.n,self.ose.n+len(new_nodes),1)]
                
                self.ose.add_new_nodes_and_lstsq_embed(hor_entries=new_rows, ver_entries= new_cols,
                                                       hor_M=hor_M, ver_M=ver_M)

            ###### lost nodes ######
            lost_nodes = list(set(self.nodes_in_order) - set(nodes_new_G))
            if len(lost_nodes)>0:
                # I lost some nodes
                if self.verbose:
                    print(f"From the original {original_n} nodes you've lost {len(lost_nodes)} nodes.")
                lost_indices = np.array([(n in lost_nodes) for n in self.nodes_in_order])
                lost_indices = lost_indices.nonzero()[0]
                
                self.ose.remove_nodes(lost_indices)
                # I remove the nodes from the list
                # TODO there's surely a better way
                for lost_node in lost_nodes:
                    self.nodes_in_order.remove(lost_node)
                
            # any remaining differences are corrected by this last step
            adj_matrix_new = nx.to_numpy_array(new_G, nodelist=self.nodes_in_order)
            delta_M = None
            if self.unknown_edge_attr is not None:
                new_mask = nx.to_numpy_array(new_G, nodelist=self.nodes_in_order, weight=self.unknown_edge_attr)
                new_M  = np.ones_like(new_mask) - np.eye(new_mask.shape[0]) - new_mask
                delta_M = new_M - self.ose.M
                
            delta_adj = adj_matrix_new - self.ose.adj_matrix
            self.ose.update_adj_matrix(delta_adj, delta_M)

            
