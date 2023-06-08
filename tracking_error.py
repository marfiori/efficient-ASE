#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracking_error -- Compute ASE-estimation error for gradient-based embeddings vs. the method from Brand, Matthew. "Fast low-rank modifications of the thin singular value decomposition." Linear algebra and its applications  (2006).


Created on Jun 20, 2022

@author: flarroca
"""

import numpy as np
import matplotlib.pyplot as plt
from graspologic.simulations import rdpg
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import sys
from out_of_sample_extension.out_of_sample_extension_np import OutOfSampleExtensionNp
import os
import traceback
from tqdm import trange


__all__ = []
__version__ = 0.2
__date__ = '2022-06-20'
__updated__ = '2023-06-08'

plt.close(fig='all')
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 28
plt.rcParams['text.latex.preamble'] = r"\usepackage{bm}" +r"\usepackage{amsfonts}" 
plt.rcParams['lines.markersize'] = 15
plt.rcParams['axes.grid'] = True


def main(argv=None):
    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    program_license = '''%s

  Created by Federico La Rocca on %s.
  Copyright (C) 2023 Marcelo Fiori, Federico La Rocca, Bernardo Marenco.

  This file is part of efficient-ASE.

  efficient-ASE is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

  efficient-ASE is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with efficient-ASE. If not, see <https://www.gnu.org/licenses/>. 

USAGE
''' % (program_shortdesc, str(__date__))

    try:
        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="enable verbose mode")
        parser.add_argument('-V', '--version', action='version', version=program_version_message)
        parser.add_argument("-n", "--n_nodes", dest="n", type=int, default=200, action="store", help="Number of nodes in the graph")
        parser.add_argument("-r", "--runs", dest="n_runs", type=int, default=10, action="store", help="Number of times to repeat the error calculation")
        parser.add_argument("-o", "--out_path", metavar="out_path", type=str,default=None, action="store", help="Path to store error plots")
        parser.add_argument("-p", "--plot_embeddings", dest="plot_embeddings", help="whether to plot the embeddings at each timestep",action="store_true")
        
        
        # Process arguments
        args = parser.parse_args()

        n = args.n
        n_runs = args.n_runs
        verbose = args.verbose
        out_path = args.out_path
        plot_embeddings = args.plot_embeddings
        
        if verbose:
            print("Verbose mode on")
            
            
        X_original_group1 = np.hstack((np.ones((int(0.7*n),1))*np.sqrt(0.3),np.zeros((int(0.7*n),1))))
        X_original_group2 = np.hstack((np.ones((int(0.3*n),1))*np.sqrt(0.8),np.zeros((int(0.3*n),1))))
        theta = np.radians(70)
        r = np.array(( (np.cos(theta), -np.sin(theta)),
                       (np.sin(theta),  np.cos(theta)) ))
        X_original = np.vstack((X_original_group1,X_original_group2@r))
        
        X_final_grupo1 = np.hstack((np.ones((int(0.3*n),1))*np.sqrt(0.8),np.zeros((int(0.3*n),1))))
        X_final_grupo2 = np.hstack((np.ones((int(0.7*n),1))*np.sqrt(0.3),np.zeros((int(0.7*n),1))))
        theta = np.radians(70)
        r = np.array(( (np.cos(theta), -np.sin(theta)),
                       (np.sin(theta),  np.cos(theta)) ))
        X_final = np.vstack((X_final_grupo1,X_final_grupo2@r))
        
        indices = np.arange(n)
        
        # Save mean square error for both methods
        mse_lwi_array = np.zeros((n_runs,n))
        mse_gd_array = np.zeros((n_runs,n))
        
        for j in range(n_runs):
            if verbose:
                print(f"Run: {j+1}")
            
            G_original = rdpg(X_original)
            
            oos_lwi = OutOfSampleExtensionNp(G_original, d=2, initial_embedding_method='gd')
            oos_gd = OutOfSampleExtensionNp(G_original, d=2, initial_embedding_method='gd')
            
            np.random.shuffle(indices)
            X_evol = X_original
            
            for i in trange(n, desc="Node change error", unit="node changes"):
                
                node = indices[i]
                
                X_evol[node,:] = X_final[node,:]
                new_adj = rdpg(X_evol)
                
                new_row = new_adj[node,:]
                new_col = new_adj[:,node]
                
                new_adj_matrix = oos_lwi.adj_matrix.copy() 
                new_adj_matrix[node,:] = new_row
                new_adj_matrix[:,node] = new_col
                    
                delta_adj = new_adj_matrix-oos_lwi.adj_matrix
                
                oos_lwi.update_adj_matrix_lwi_svd(delta_adj)
                oos_gd.update_adj_matrix(delta_adj)
                    
                P = X_evol@X_evol.T
                mse_lwi = np.linalg.norm(((oos_lwi.Xhat@oos_lwi.Xhat.T)-P)*oos_lwi.M,ord='fro')
                mse_gd = np.linalg.norm(((oos_gd.Xhat@oos_gd.Xhat.T)-P)*oos_gd.M,ord='fro')
                
                mse_lwi_array[j,i] = mse_lwi
                mse_gd_array[j,i] = mse_gd
                
                if plot_embeddings and out_path is not None:
                    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(15,10))
                        
                    ax1.set_xlim([-1.1, 1.1])
                    ax1.set_ylim([-1, 1])
                    
                    ax1.scatter(oos_gd.Xhat[:,0],oos_gd.Xhat[:,1],color='maroon') 
                    ax1.set_title('Gradient Descent',fontsize=36)
                    ax1.grid()
                    
                
                    ax2.scatter(oos_lwi.Xhat[:,0],oos_lwi.Xhat[:,1],color='darksalmon') 
                    ax2.set_title('[Brand \'06]',fontsize=36)
                    ax2.grid()
                    
                    fig.subplots_adjust(wspace=0.045,left=0.08,right=0.98,top=0.86,bottom=0.06)
                    fig.suptitle(f'$\hat{{\mathbf{{X}}}}_t$ at $t = {i}$. ',fontsize=40)
                    plt.savefig(out_path+'change_'+str(i)+'.png',format='png')
                    plt.savefig(out_path+'change_'+str(i)+'.pdf',format='pdf')
                    plt.close(fig=fig)
                    
        mse_gd_quartiles = np.quantile(mse_gd_array,q=[0.25,0.5,0.75],axis=0)
        mse_lwi_quartiles = np.quantile(mse_lwi_array,q=[0.25,0.5,0.75],axis=0)
        
        changed_nodes = np.arange(1,n+1)
        
        fig,ax = plt.subplots(figsize=(20,10))
        ax.plot(changed_nodes,mse_lwi_quartiles[1],label='[Brand \'06]',color='darksalmon')
        ax.fill_between(changed_nodes, mse_lwi_quartiles[0], mse_lwi_quartiles[2], alpha=0.3, facecolor='darksalmon')
        ax.plot(changed_nodes,mse_gd_quartiles[1],label='Gradient Descent',color='maroon')
        ax.fill_between(changed_nodes, mse_gd_quartiles[0], mse_gd_quartiles[2], alpha=0.3, facecolor='maroon')
        ax.set_xlabel(r'\# of nodes that changed',fontsize=32)
        fig.suptitle(r'Evolution of $||\hat{\mathbf{X}}_t\hat{\mathbf{X}}_t^\top - \mathbf{P}_t||_F$',fontsize=40)
        ax.legend(fancybox=True, shadow=True,fontsize=32)
        fig.subplots_adjust(left=0.04,right=0.98,top=0.91,bottom=0.11)
        
        if out_path is not None:
            plt.savefig(out_path+'error_brand.png',format='png')
            plt.savefig(out_path+'error_brand.pdf',format='pdf')
            plt.close(fig)
        else:
            plt.show()
            
        return 0
    
    except Exception as e:
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use --help\n\n")
        print(traceback.format_exc())
        return 2
    

if __name__ == "__main__":
    main()
