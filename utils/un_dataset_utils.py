'''
Created on May 18, 2023

@author: bernardo
'''
import os
import requests
from tqdm import tqdm
import pandas as pd
import pycountry_convert as pc
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

resolutions_issues = {'me': 'Palestinian conflict', 
                      'nu': 'Nuclear weapons and nuclear material', 
                      'di': 'Arms control and disarmament',
                      'co': 'Colonialism',
                      'hr': 'Human rights',
                      'ec': 'Economic Development',
                      'N/A': 'Not specified'}

resolutions_issues_color = {'me': 'salmon', 
                            'nu': 'yellow', 
                            'di': 'teal',
                            'co': 'orchid',
                            'hr': 'navy',
                            'ec': 'orange',
                            'N/A': 'black'}

continents_colors = {'North America': 'yellow',
                     'South America': 'forestgreen',
                     'Europe': 'royalblue',
                     'Africa': 'plum',
                     'Asia': 'darkorange',
                     'Oceania': 'firebrick'}


def load_un_dataset(un_data_path, initial_year=1946, final_year=2018, remove_nonmembers=True, remove_nonpresent=False, unknown_votes=False):
    
    if os.path.isdir(os.path.dirname(un_data_path)):
        if not os.path.exists(un_data_path):
            download_un_dataset(un_data_path)
    else:
        raise Exception("Provided path for UN dataset is not reachable")
    
    # Load data    
    votes_df = pd.read_csv(un_data_path, low_memory=False)
    # Keep only desired years
    votes_df = votes_df[votes_df.year>=initial_year]
    votes_df = votes_df[votes_df.year<=final_year]
    
    if remove_nonmembers:
        # Remove votes by nonmembers
        votes_df = votes_df[votes_df.vote!=9]
    
    if remove_nonpresent:
        # Remove votes by nonmembers
        votes_df = votes_df[votes_df.vote!=8]
        
    # Edges in graph represent an affirmative vote
    votes_df['weight'] = (votes_df.vote==1)
    
    if unknown_votes:
        # Voters preference is assumed unknown if it is an abstention or voter is not present
        votes_df['unknown'] =  (votes_df.vote==2) | (votes_df.vote==8)
        
    return votes_df

        
def download_un_dataset(filename='UNVotes-1.csv', data_url='https://dataverse.harvard.edu/api/access/datafile/6358426'):
    # Code from https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
    response = requests.get(data_url, stream=True)
    total_size = int(response.headers.get('content-length',0))
    with open(filename, "wb") as f, tqdm(desc='Downloading UN dataset', total=total_size, unit='B', unit_divisor=1024, unit_scale=True) as pbar:
        for un_data in response.iter_content(chunk_size=1024):
            size = f.write(un_data)
            pbar.update(size)
            
def get_continents_dict(votes_df):
    continents_dict = {}
    countries = votes_df.Country.unique()
    for country in countries: 
        try:
            continent_code = pc.country_alpha2_to_continent_code(pc.country_alpha3_to_country_alpha2(country))
            continents_dict[country] = pc.convert_continent_code_to_continent_name(continent_code)
        except:
            continue
            # print(pais)
            
    continents_dict['DDR'] = 'Europe'
    continents_dict['CSK'] = 'Europe'
    continents_dict['YUG'] = 'Europe'
    continents_dict['EAZ'] = 'Africa'
    continents_dict['YAR'] = 'Asia'
    continents_dict['TLS'] = 'Asia'
    
    return continents_dict

def get_countries_name_conversion_dict(votes_df):
    countries = votes_df.Countryname.unique()
    conversion_dict = {}
    for country in countries: 
        conversion_dict[country] = votes_df[votes_df.Countryname==country].Country.unique()[0]
        
    return conversion_dict

def create_un_graphs(votes_df):
    
    continents_dict = get_continents_dict(votes_df)
    conversion_dict = get_countries_name_conversion_dict(votes_df)
    
    all_graphs = []
    years = []
    
    initial_year = votes_df.year.min()
    final_year = votes_df.year.max()
    
    edge_attr = ['weight', 'unknown'] if 'unknown' in votes_df.columns else 'weight'    
    
    for year in range(initial_year,final_year+1):
        
        votes_df_year= votes_df[votes_df.year==year]
        
        g = nx.from_pandas_edgelist(votes_df_year,source='Countryname',target='resid',edge_attr=edge_attr,create_using=nx.DiGraph())
        if g.number_of_edges()>0:
            
            countries_list = votes_df_year.Countryname.unique()
            
            # Add country's code and continent as graph attributes
            countries_codes = {}
            countries_continents = {}
            nodes_colors = {}
            for country in countries_list:
                countries_codes[country] = conversion_dict[country]
                countries_continents[country] = continents_dict[conversion_dict[country]]
                nodes_colors[country] = continents_colors[countries_continents[country]]
                
            nx.set_node_attributes(g, countries_codes, name='country code')
            nx.set_node_attributes(g, countries_continents, name='continent')
            
            
            # Add resolution's issue as graph attribute
            resolutions_list = votes_df_year.resid.unique()
            resolutions_issues_dict = {}
            important_resolutions_dict = {}
            
            for resolution_id in resolutions_list:
                df_votes_sum = votes_df_year[votes_df_year.resid==resolution_id][['me','nu','di','co','hr','ec']].sum()
                if df_votes_sum.max()>0:
                    resolutions_issues_dict[resolution_id] = df_votes_sum.idxmax()
                else:
                    resolutions_issues_dict[resolution_id] = 'N/A'
                    
                nodes_colors[resolution_id] = resolutions_issues_color[resolutions_issues_dict[resolution_id]]
                
                important_vote = votes_df_year[votes_df_year.resid==resolution_id]['importantvote'].max()
                if important_vote > 0:
                    important_resolutions_dict[resolution_id] = True
                else:
                    important_resolutions_dict[resolution_id] = False
            
            nx.set_node_attributes(g, resolutions_issues_dict,name='issue code')
            nx.set_node_attributes(g, nodes_colors, name='color')
            nx.set_node_attributes(g, important_resolutions_dict, name='important vote')
                
            all_graphs.append(g)
            years.append(year)
            
    return all_graphs, years

def plot_embeddings(graph,Xl,Xr,nodes_in_order,countries_to_show=None, countries_colors=None, plot_resolutions=True, dropna_resolutions=True, ax=None):
    
    if ax is None:
        _, ax = plt.subplots(figsize=[12,6],nrows=1,ncols=1)
        
    # Get countries and resolutions list
    countries = [country for country, code in graph.nodes.data('country code', default=None) if code!=None]
    
    # Get countries idxs as they appear in Xl and Xr 
    countries_idxs = [nodes_in_order.index(country) for country in countries]

    # Get nodes color from graph attributes
    nodes_colors = np.array([graph.nodes[node]['color'] for node in nodes_in_order])
    
    
    if plot_resolutions:
        if dropna_resolutions:
            resolutions = {resolution_id:issue_code for resolution_id, issue_code in graph.nodes.data('issue code', default='N/A') if issue_code!='N/A'}
        else:
            resolutions = {resolution_id:issue_code for resolution_id, issue_code in graph.nodes.data('issue code', default='None') if issue_code!='None'}

        # Get resolutions idxs as they appear in Xl and Xr 
        resolutions_idxs = np.array([nodes_in_order.index(resolution_id) for resolution_id in resolutions])
        
        ax.scatter(Xr[resolutions_idxs,0],Xr[resolutions_idxs,1],c=nodes_colors[resolutions_idxs],alpha=1,marker='x')
        
        # Get issues list for this graph and add them to plot legend
        issues_list = np.unique(list(resolutions.values()))
        legend_elements = []
        
        for issue in issues_list:
            legend_elements.append(Line2D([0], [0], marker='x', linestyle='none',
                                          color=resolutions_issues_color[issue],
                                          label=resolutions_issues[issue],
                                          markerfacecolor=resolutions_issues_color[issue],
                                          markeredgewidth=3,
                                          markersize=15))
            
        ax.legend(handles=legend_elements,loc='lower left',fancybox=True, shadow=True, ncol=2, fontsize=26, handletextpad=0.01,columnspacing=0.1,borderpad=0.1)
        
    if countries_to_show is not None:
        for i,country in enumerate(countries_to_show):
            if country in countries:
                country_idx = nodes_in_order.index(country)
                countries_idxs.remove(country_idx)
                ax.annotate(graph.nodes[country]['country code'], (Xl[country_idx,0],Xl[country_idx,1]),color=countries_colors[i])
                ax.scatter(Xl[country_idx,0],Xl[country_idx,1],c=countries_colors[i],marker='o')
                
        ax.scatter(Xl[countries_idxs,0],Xl[countries_idxs,1],c='lightgray',alpha=0.3,marker='o')
    else:
        ax.scatter(Xl[countries_idxs,0],Xl[countries_idxs,1],c=nodes_colors[countries_idxs],alpha=0.3,marker='o')
    
    return ax
    