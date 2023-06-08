#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
node_addition_cumulative_error -- Compute ASE-estimation error for gradient-based embeddings vs. the method from Levin, Keith, et al. "Out-of-sample extension of graph adjacency spectral embedding" - ICML 2018
    
Created on Jun 20, 2022

@author: flarroca
'''

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from out_of_sample_extension.out_of_sample_extension_np import OutOfSampleExtensionNp
from tqdm import trange
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import sys
import traceback
import os

__all__ = []
__version__ = 0.1
__date__ = '2022-06-20'
__updated__ = '2023-06-08'

plt.close(fig='all')
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 32
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
        parser.add_argument("-n", "--n_new", dest="n_new", type=int, default=1000, action="store", help="Number of new nodes to add")
        parser.add_argument("-r", "--runs", dest="n_runs", type=int, default=10, action="store", help="Number of times to repeat the error calculation")
        parser.add_argument("-o", "--out_path", metavar="out_path", type=str,default=None, action="store", help="Path to store error plot")



        # Process arguments
        args = parser.parse_args()

        n_new = args.n_new
        n_runs = args.n_runs
        verbose = args.verbose
        out_path = args.out_path

        if verbose:
            print("Verbose mode on")
            
        # Erdos-Renyi graph parameters
        n = 100
        p = 0.1
        
        # Save mean square error for both methods
        mse_projection_array = np.zeros((n_runs,n_new))
        mse_gd_array = np.zeros((n_runs,n_new))
        
        for j in range(n_runs):
            if verbose:
                print(f"Run: {j+1}")
            
            G = nx.erdos_renyi_graph(n,p)
            A = nx.to_numpy_array(G)
            
            oos_projection = OutOfSampleExtensionNp(A, d=1)
            oos_gd = OutOfSampleExtensionNp(A, d=1)
            
        
            for i in trange(n_new, desc="Node addition error", unit="new node"):
                
                P = p*np.ones((n+i+1,n+i+1))
                new_row = np.random.binomial(n=1,p=p,size=(1,n+i))
                oos_projection.add_one_node_and_lstsq_embed(new_row)
                hor_entries = oos_projection.adj_matrix[n+i:,:]
                ver_entries = oos_projection.adj_matrix[:,n+i:]
                oos_gd.add_new_nodes(hor_entries, ver_entries)
                mse_projection = np.linalg.norm(((oos_projection.Xhat@oos_projection.Q@oos_projection.Xhat.T)-P)*oos_projection.M,ord='fro')/np.sqrt(n+i+1)
                mse_gd = np.linalg.norm(((oos_gd.Xhat@oos_gd.Q@oos_gd.Xhat.T)-P)*oos_gd.M,ord='fro')/np.sqrt(n+i+1)
                
                mse_gd_array[j,i] = mse_gd
                mse_projection_array[j,i] = mse_projection
        
        
        mse_gd_quartiles = np.quantile(mse_gd_array,q=[0.25,0.5,0.75],axis=0)
        mse_projection_quartiles = np.quantile(mse_projection_array,q=[0.25,0.5,0.75],axis=0)
        
        new_nodes = np.arange(1,n_new+1)
        
        fig,ax = plt.subplots(figsize=(20,10))
        ax.plot(new_nodes,mse_projection_quartiles[1],label='[Levin et al. \'18]',color='darksalmon')
        ax.fill_between(new_nodes, mse_projection_quartiles[0], mse_projection_quartiles[2], alpha=0.3, facecolor='darksalmon')
        ax.plot(new_nodes,mse_projection_quartiles[1],label='Gradient Descent',color='maroon')
        ax.fill_between(new_nodes, mse_gd_quartiles[0], mse_gd_quartiles[2], alpha=0.3, facecolor='maroon')
        ax.set_xlabel(r'\# of new nodes',fontsize=32)
        fig.suptitle(r'Evolution of $||\hat{\mathbf{X}}_t\hat{\mathbf{X}}_t^\top - \mathbf{P}_t||_F/\sqrt{N_t}$',fontsize=40)
        ax.legend(fancybox=True, shadow=True,fontsize=32)
        fig.subplots_adjust(left=0.05,right=0.99,top=0.9,bottom=0.1)
        
        if out_path is not None:
            plt.savefig(out_path+'.png',format='png')
            plt.savefig(out_path+'.pdf',format='pdf')
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