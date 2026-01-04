#!/usr/bin/env python3

import sys
print(sys.version)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import isfile
from scipy import stats as stats
import pandas as pd
from matplotlib.lines import Line2D
import parameters as param
from functions.general_functions import hist_normalised
from collections import Counter

plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[cm]{sfmath}'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'cm'

K_Dmap, L_Dmap = 4, 8
n_list = (50, 100, 500)
###
color_list = matplotlib.cm.tab10(range(10))[2:2+len(n_list)]
legend_elements = [Line2D([0], [0], mfc=color_list[i], ls='', marker='o', label='$n_p$='+str(N_pheno), mew=0, ms=5) for i, N_pheno in enumerate(n_list)]
###############################################################################################
print('\n\nplot data from deterministic version of synthetic GP map with changes in parameters', flush=True)
###############################################################################################
f, ax = plt.subplots(ncols=6, figsize=(12, 2), width_ratios=[1,]*5 +[0.5])
N_vs_fit_log = {}
for i, N_pheno in enumerate(n_list):
   description_parameters = 'synthetic_normal_' + str(N_pheno)+'_'+ str(L_Dmap) + '_' + str(K_Dmap) 
   DGPmapdata_filename, DGrobustness_filename, DGevolvability_filename = './data/GPmapproperties_'+description_parameters+'.csv', './data/Grobustness_'+description_parameters+'.npy', './data/Gevolvability_'+description_parameters+'.npy'
   try:
      df_DGPmap = pd.read_csv(DGPmapdata_filename)
   except IOError:
      print('not found', DGPmapdata_filename)
      continue
   N_list = df_DGPmap['neutral set size'].tolist()
   list_all_possible_folded_structures = df_DGPmap['phenotype'].tolist()
   rho_list = df_DGPmap['rho'].tolist()
   Dph_vs_rho = {ph: rho for ph, rho in zip(df_DGPmap['phenotype'].tolist(), df_DGPmap['rho'].tolist())}
   DPevolv_list = df_DGPmap['p_evolv'].tolist()
   N_list_nonzero = [f for f in  N_list if not np.isnan(f) and f > 0]
   DGrobustness =  np.load(DGrobustness_filename)
   DGevolvability = np.load(DGevolvability_filename)
   ######
   rank_data, N_data_by_rank = np.arange(1, len(N_list_nonzero) + 1), sorted(N_list_nonzero, reverse=True)
   ax[0].scatter(rank_data, np.divide(N_data_by_rank, K_Dmap**L_Dmap), color=color_list[i], s=4, lw = 0)
   ax[0].set_ylabel(r'$f_p$', fontsize=14)
   ax[0].set_xlabel('freq. rank', fontsize=13)
   ax[0].set_yscale('log')
   ax[0].set_xscale('log')
   ######
   ax[1].scatter(np.log10([N/K_Dmap**L_Dmap for i, N in enumerate(N_list) if not np.isnan(N) and N > 0]), [rho for i, rho in enumerate(rho_list) if not np.isnan(N_list[i]) and N_list[i] > 0], color=color_list[i], s=4, lw = 0)
   ax[1].set_ylabel(r'$\rho_p$', fontsize=14)
   ax[1].set_xlabel(r'$\log_{10} f_p$', fontsize=13)
   slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10([f/K_Dmap**L_Dmap for f, rho in zip(N_list, rho_list) if not np.isnan(f) and not np.isnan(rho) and f > 0]), 
                                                                  [rho for f, rho in zip(N_list, rho_list) if not np.isnan(f) and not np.isnan(rho) and f > 0])
   N_vs_fit_log[N_pheno] = (slope, intercept)
   N_corr_outlisers = [f for f, rho in zip(N_list, rho_list) if not np.isnan(f) and f > 0 and f/K_Dmap**L_Dmap >= rho]
   if len(N_corr_outlisers) > 0:
      print('D GP map', description_parameters, 'phenotypes with f >= rho', len(N_corr_outlisers), 'out of', len(N_list_nonzero) , 'with neutral set size', Counter(N_corr_outlisers))
      print('max robustness of corr outliers', max([rho for f, rho in zip(N_list, rho_list) if not np.isnan(f) and f > 0 and f/K_Dmap**L_Dmap >= rho]))
   ######
   assert abs(i - 1) <= 1
   ax[2].scatter(DGrobustness.flatten() + (i-1) * 0.1/((K_Dmap-1)*L_Dmap), DGevolvability.flatten(), s=0.5, color=color_list[i])
   max_gevolv = max(DGevolvability.flatten())
   ax[2].set_ylabel(r'$\epsilon_g$', fontsize=14)
   ax[2].set_xlabel(r'$\rho_g$', fontsize=14)
   for grho, gev in zip(DGrobustness.flatten(), DGevolvability.flatten()):
      assert (np.isnan(gev) or np.isnan(grho)) or int(gev) <= int(round((1-grho)* (K_Dmap-1)*L_Dmap))# + 0.01 
   ######
   ax[3].scatter(rho_list, np.array(DPevolv_list)/(len(N_list_nonzero) - 1), s=3, color=color_list[i], lw = 0)
   ax[3].set_ylabel(r'$\epsilon_p$  (normalised)', fontsize=13)
   ax[3].set_xlabel(r'$\rho_p$', fontsize=14)
   ax[3].set_xlim(-0.05, 1)
   assert np.nanmin(rho_list) >= -0.05 and np.nanmax(rho_list) <= 1
   ######
   df_shapespace = pd.read_csv('./data/shape_space_covering'+description_parameters+'.csv')
   ax[4].plot(df_shapespace['dist'].tolist(), np.array(df_shapespace['mean # pheno']), color=color_list[i])
   ax[4].set_xlabel(r'Hamming dist. $d$', fontsize=13)
   ax[4].set_ylabel('mean fraction\nof phenos\n'+r'found within $d$', fontsize=13)
   ###
   if len([i for i, N in enumerate(N_list) if N > 0 and rho_list[i] == 0]):
         print('not shown non-zero-freq/norobustness datapoints', len([i for i, N in enumerate(N_list) if N > 0 and rho_list[i] == 0]), 'out of', len(rho_list))
ax[-1].legend(handles=legend_elements)
xlims = ax[1].get_xlim()
x_values = np.linspace(xlims[0], xlims[1], num=500)
ax[1].plot(x_values, np.power(10, x_values), c='k', ls=':')
ax[1].set_xlim(xlims)
for i, N_pheno in enumerate(n_list):
   slope, intercept = N_vs_fit_log[N_pheno]
   ax[1].plot([x for x in x_values if x * slope + intercept > 0], [x for x in slope*x_values + intercept if x > 0], color=color_list[i], lw=0.5, zorder=-3) 
ax[-1].axis('off')
for i in  range(5):
   ax[i].annotate('ABCDEF'[i], xy=(0.06, 1.1), xycoords='axes fraction', fontsize=12, weight='bold')
f.tight_layout()
f.savefig('./plots/DGPmap_analysis'+'.png', bbox_inches='tight', dpi=250)
plt.close('all')

###############################################################################################
print('\n\nplot versatility for deterministic version of synthetic GP map ', flush=True)
###############################################################################################
for i, N_pheno in enumerate(n_list):
   color = color_list[i]
   description_parameters = 'synthetic_normal_' + str(N_pheno)+'_'+ str(L_Dmap) + '_' + str(K_Dmap) 
   f, ax = plt.subplots(ncols = 3, figsize=(9, 2.2))
   try:
      df_versatility = pd.read_csv('./data/DGPmap_versatility'+description_parameters+'.csv')
   except IOError:
      continue
   rank_list = [c.split('rank')[-1].strip() for c in df_versatility.columns if c.startswith('mean')]
   for i, rank in enumerate(rank_list):
      ax[i].errorbar(np.arange(L_Dmap) + 1,  np.array(df_versatility['mean v  - rank '+str(rank)].tolist()), yerr=np.array(df_versatility['std v - rank '+str(rank)].tolist()), marker='o', ms=3, capsize=3, color='grey')
      ax[i].set_xlabel('position', fontsize=13)
      ax[i].set_ylabel('number of\nneutral mutations', fontsize=13)  
      ax[i].plot([0, L_Dmap+1], [K_Dmap-1, K_Dmap-1], c='k')     
      ax[i].plot([0, L_Dmap+1], [0, 0], c='k')    
      ax[i].set_xlim(0.5, L_Dmap + 0.5) 
      ax[i].set_xticks([i for i in np.arange(1, 0.5 + L_Dmap)]) 
      assert min(np.arange(L_Dmap) + 1) >= 0.5 and max(np.arange(L_Dmap) + 1) <= L_Dmap 
      ax[i].set_title('phenotype rank ' + str(rank.split(' ')[0]) + ', NC size ' + str(rank.split(' ')[-1]))
   f.tight_layout()
   f.savefig('./plots/DGPmap_versatility'+description_parameters+'.png', dpi=200, bbox_inches='tight')
   plt.close('all')
   ############
   f, ax = plt.subplots(nrows = 3, ncols=L_Dmap, figsize=(L_Dmap*3, 9))
   for i, rank in enumerate(rank_list):
      pos_vs_hist = {pos: [] for pos in range(L_Dmap)}
      for pos in range(L_Dmap):
         hist = [df_versatility[str(v) + 'neutral prevalance - rank '+str(rank)].tolist()[pos] for v in range(K_Dmap)]
         assert sum(hist) == int(rank.split(' ')[-1]) #sum equals NC size
         ax[i, pos].bar(np.arange(K_Dmap), hist)
         ylims = ax[i, pos].get_ylim()
         mean, std = df_versatility['mean v  - rank '+str(rank)].tolist()[pos], df_versatility['std v - rank '+str(rank)].tolist()[pos]
         assert np.abs(mean - sum([i*f for i, f in enumerate(hist)])/sum(hist)) < 0.01
         ax[i, pos].plot([mean,]*2, ylims, c='k')
         ax[i, pos].fill_between([mean - std, mean + std], 0, max(ylims), color='grey', zorder=-2)
         ax[i, pos].set_xlabel('number of neutral mutations')
         ax[i, pos].set_title('pos='+str(pos))
         if pos == L_Dmap//2:
            ax[i, pos].set_title('phenotype rank ' + str(rank.split(' ')[0]) + ', NC size ' + str(rank.split(' ')[-1])) 
         ax[i, pos].set_xlim(-0.5, K_Dmap -0.5) 
   f.tight_layout()
   f.savefig('./plots/DGPmap_versatility'+description_parameters+'_2.png', dpi=200, bbox_inches='tight')
   plt.close('all')
   ############
   f, ax = plt.subplots(figsize=(3, 2.2))
   df_NCs = pd.read_csv('./data/DGPmap_nNCs'+description_parameters+'.csv')
   hist_normalised(df_NCs['# NCs'].tolist(), bins=np.arange(0.5, max(df_NCs['# NCs'].tolist())+0.6), ax=ax, color='grey')
   ax.set_xlabel('number NCs per phenotype', fontsize=13)
   ax.set_xticks(np.arange(1, max(df_NCs['# NCs'].tolist()) + 0.5))
   ax.set_ylabel('fraction of phenotypes', fontsize=13)
   print('have #NC data for ', len([x for x in df_NCs['# NCs'].tolist() if x >= 1]))
   f.tight_layout()
   f.savefig('./plots/DGPmap_NC_fragmentation'+description_parameters+'.png', dpi=200, bbox_inches='tight')
   plt.close('all')