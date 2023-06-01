#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on May 2, 2023

@author: bernardo
'''

import matplotlib.pyplot as plt
import os
from utils.un_dataset_utils import load_un_dataset, create_un_graphs, plot_embeddings

from out_of_sample_extension.out_of_sample_extension_nx import OutOfSampleExtensionNx
import numpy as np
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import sys
import traceback

__all__ = []
__version__ = 0.1
__date__ = '2023-06-01'
__updated__ = '2023-06-01'

plt.close(fig='all')
plt.rcParams['lines.linewidth'] = 3
#plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 28
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

  Created by Bernardo Marenco on %s.
  Copyright 2023 organization_name. All rights reserved.

  This file is part of Foobar.

  Foobar is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

  Foobar is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with Foobar. If not, see <https://www.gnu.org/licenses/>. 

USAGE
''' % (program_shortdesc, str(__date__))

    try:
        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="enable verbose mode")
        parser.add_argument('-V', '--version', action='version', version=program_version_message)
        parser.add_argument(dest="un_data_path", help="path to UN dataset file. If file does not exist, but the path is reachable, the file will be downloaded", metavar="path")
        parser.add_argument("-i", "--initial_year", dest="initial_year", type=int, default=1946, action="store", help="Initial year to consider in the UN dataset file")
        parser.add_argument("-f", "--final_year", dest="final_year", type=int, default=2018, action="store", help="Final year to consider in the UN dataset file")
        parser.add_argument("-d", "--dimension", dest="d", type=int, default=2, action="store", help="Embeddings dimension")
        parser.add_argument("-c", "--countries", dest="which_countries", type=str, nargs='+', action="store", help="List of countries to highlight in the embeddings' plots", default=None)
        parser.add_argument("-o", "--out_path", metavar="out_path", type=str,default=None, action="store", help="path to folder where plots are to be saved")
        parser.add_argument("-u", "--unknown_votes", dest="unknown_votes", help="whether to consider absent votes as unknowns",action="store_true")



        # Process arguments
        args = parser.parse_args()

        un_data_path = args.un_data_path
        initial_year = args.initial_year
        final_year = args.final_year
        d = args.d
        verbose = args.verbose
        unknown_votes = args.unknown_votes
        out_path = args.out_path
        which_countries = args.which_countries

        if verbose:
            print("Verbose mode on")
            print('Creating UN dataset...')
            
        votes_df = load_un_dataset(un_data_path, initial_year=initial_year, final_year=final_year,
                                   remove_nonmembers=True, remove_nonpresent=False, unknown_votes=unknown_votes)
    
        all_graphs, years = create_un_graphs(votes_df)  
        
        if verbose:
            print('Done creating UN dataset')
    
        colors = ['royalblue','firebrick','forestgreen','olive']
        
        
        initial_graph = all_graphs[0]
        if unknown_votes:
            oos = OutOfSampleExtensionNx(initial_graph, initial_embedding_method='gd',
                                         unknown_edge_attr='unknown', d=d,
                                         max_iter=5000, b=0.2, sigma=0.1, t=0.1,
                                         verbose=verbose)
        else:
            oos = OutOfSampleExtensionNx(initial_graph, initial_embedding_method='gd',
                                         unknown_edge_attr=None, d=d,
                                         max_iter=1000, b=0.3, sigma=0.1, t=0.1,
                                         verbose=verbose)
        
        if which_countries is not None:
            countries_alignment = {country:{} for country in which_countries if country!='United States of America'}
        
        for graph, year in zip(all_graphs,years):
            if verbose:
                print('***********************')
                print(f'Year: {year}')
            
            # Update embeddings
            oos.new_graph(graph)
            Xl,Xr = oos.get_embeddings()
            
            # Plot embeddings
            fig, ax = plt.subplots(figsize=[10,10])
            
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            
            
            ax = plot_embeddings(graph,Xl,Xr,oos.nodes_in_order,countries_to_show=which_countries,
                                  countries_colors=colors,ax=ax, plot_resolutions=False)
            ax.set_title(f'{year}')
            
            if which_countries is not None:
                nodes = oos.nodes_in_order
                US_idx = nodes.index('United States of America')
                Xl_us = Xl[US_idx,:]
                
                for country in countries_alignment:
                    if country in nodes:
                        test_country_idx = nodes.index(country)
                        Xl_test_country = Xl[test_country_idx,:]
                        alignment = np.dot(Xl_us,Xl_test_country)/(np.linalg.norm(Xl_us)*np.linalg.norm(Xl_test_country))
                        countries_alignment[country][year] = alignment
            if out_path is not None:
                plt.savefig(out_path+str(year)+'.png',format='png')
                plt.close()
            else:
                plt.show()
        
        if which_countries is not None:    
        
            fig_angles, ax_angles = plt.subplots(figsize=[12,6],nrows=1,ncols=1)
            
            for country in countries_alignment:
                ax_angles.plot(countries_alignment[country].keys(),countries_alignment[country].values(),label=country, linestyle='dashed')
            
            fig_angles.suptitle('Alignment with US')
            fig_angles.subplots_adjust(left=0.07,right=0.97,top=0.94,bottom=0.09,hspace=0.06)
            ax_angles.legend(loc='lower left',fancybox=True, shadow=True, ncol=3, fontsize=26, handletextpad=0.01,columnspacing=0.1,borderpad=0.1)
            
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


