#!/usr/bin/env python3

import sys
print(sys.version)
import numpy as np
from functions import synthetic_model_functions as synthetic
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import isfile, realpath
from scipy import stats as stats
from math import comb
import pandas as pd
from matplotlib.lines import Line2D
import parameters as param
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from functions.general_functions import *
from collections import Counter

plt.rcParams["text.usetex"] = True

filepath = './'
K, L = param.synthetic_K, param.synthetic_L
type_plot_vs_n_and_kbT_list = {'vary_N_pheno': [(N_pheno, 'mean_energygap') for N_pheno in (50, 100, 500)], 'vary_T': [(100, kbT) for kbT in (0.05, 0.1, 0.5, 1)]}
###
type_plot_vs_color_list = {'vary_N_pheno': [plt.get_cmap('hot')(i) for i in np.linspace(0, 0.8, len(type_plot_vs_n_and_kbT_list['vary_N_pheno']))], 'vary_T': [plt.get_cmap('cool')(i) for i in np.linspace(0, 1, len(type_plot_vs_n_and_kbT_list['vary_T']))]}
legend_elements = {'vary_N_pheno': [Line2D([0], [0], mfc=type_plot_vs_color_list['vary_N_pheno'][i], ls='', marker='o', label='$n_p$='+str(N_pheno), mew=0, ms=5) for i, (N_pheno, kbT) in enumerate(type_plot_vs_n_and_kbT_list['vary_N_pheno'])],
                  'vary_T': [Line2D([0], [0], mfc=type_plot_vs_color_list['vary_T'][i], ls='', marker='o', label='$T$='+str(kbT), mew=0, ms=5) for i, (N_pheno, kbT) in enumerate(type_plot_vs_n_and_kbT_list['vary_T'])]}
legend_elements_with_D = {k: v[:] for k, v in legend_elements.items()}
legend_elements_with_D['vary_T'] = [Line2D([0], [0], mfc='grey', ls='', marker='o', label=r'deterministic', mew=0, ms=5),] + legend_elements_with_D['vary_T'][:]
###############################################################################################
print('\n\nplot data from deterministic version of synthetic GP map with changes in parameters', flush=True)
###############################################################################################
K_Dmap, L_Dmap = 4, 9
f, ax = plt.subplots(ncols=6, figsize=(13.5, 2), width_ratios=[1,]*5 +[0.5])
type_plot = 'vary_N_pheno' #temperature has no impact on D GP map
N_vs_fit_log = {}
for i, (N_pheno, kbT) in enumerate(type_plot_vs_n_and_kbT_list[type_plot]):
   color_list = type_plot_vs_color_list[type_plot]
   description_parameters = 'synthetic_normal_' + str(N_pheno)+'_'+ str(L_Dmap) + '_' + str(K_Dmap) 
   DGPmapdata_filename, DGrobustness_filename, DGevolvability_filename = filepath + 'data/GPmapproperties_'+description_parameters+'.csv', filepath + 'data/Grobustness_'+description_parameters+'.npy', filepath + 'data/Gevolvability_'+description_parameters+'.npy'
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
   ax[0].scatter(rank_data, np.divide(N_data_by_rank, K_Dmap**L_Dmap), color=color_list[i], s=4, alpha=0.5, lw = 0)
   ax[0].set_ylabel(r'$f_p$')
   ax[0].set_xlabel('freq. rank')
   ax[0].set_yscale('log')
   ax[0].set_xscale('log')
   ######
   ax[1].scatter(np.log10([N/K_Dmap**L_Dmap for i, N in enumerate(N_list) if not np.isnan(N) and N > 0]), [rho for i, rho in enumerate(rho_list) if not np.isnan(N_list[i]) and N_list[i] > 0], color=color_list[i], s=4, alpha=0.5, lw = 0)
   ax[1].set_ylabel(r'$\rho_p$')
   ax[1].set_xlabel(r'$\log_{10} f_p$')
   slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10([f/K_Dmap**L_Dmap for f, rho in zip(N_list, rho_list) if not np.isnan(f) and not np.isnan(rho) and f > 0]), 
                                                                  [rho for f, rho in zip(N_list, rho_list) if not np.isnan(f) and not np.isnan(rho) and f > 0])
   N_vs_fit_log[N_pheno] = (slope, intercept)
   N_corr_outlisers = [f for f, rho in zip(N_list, rho_list) if not np.isnan(f) and f > 0 and f/K_Dmap**L_Dmap >= rho]
   if len(N_corr_outlisers) > 0:
      print('D GP map', description_parameters, 'phenotypes with f >= rho', len(N_corr_outlisers), 'out of', len(N_list_nonzero) , 'with neutral set size', Counter(N_corr_outlisers))
   ######
   assert abs(i - 1) <= 1
   ax[2].scatter(DGrobustness.flatten() + (i-1) * 0.1/((K_Dmap-1)*L_Dmap), DGevolvability.flatten(), s=0.5, alpha=0.8, color=color_list[i])
   max_gevolv = max(DGevolvability.flatten())
   ax[2].set_ylabel(r'$\epsilon_g$')
   ax[2].set_xlabel(r'$\rho_g$')
   for grho, gev in zip(DGrobustness.flatten(), DGevolvability.flatten()):
      assert gev <= (1-grho)* (K_Dmap-1)*L_Dmap or (np.isnan(gev) or np.isnan(grho))
   ######
   ax[3].scatter(rho_list, np.array(DPevolv_list)/(len(N_list_nonzero) - 1), s=3, color=color_list[i], alpha=0.5, lw = 0)
   ax[3].set_ylabel(r'$\epsilon_p$  (normalised)')
   ax[3].set_xlabel(r'$\rho_p$')
   ax[3].set_xlim(-0.05, 1)
   assert np.nanmin(rho_list) >= -0.05 and np.nanmax(rho_list) <= 1
   ######
   df_shapespace = pd.read_csv(filepath + 'data/shape_space_covering'+description_parameters+'.csv')
   ax[4].plot(df_shapespace['dist'].tolist(), np.array(df_shapespace['mean # pheno']), color=color_list[i], alpha=0.5)
   ax[4].set_xlabel(r'Hamming dist. $d$')
   ax[4].set_ylabel('fraction of phenos\n'+r'found within $d$')
   ###
   if len([i for i, N in enumerate(N_list) if N > 0 and rho_list[i] == 0]):
         print('not shown non-zero-freq/norobustness datapoints', len([i for i, N in enumerate(N_list) if N > 0 and rho_list[i] == 0]), 'out of', len(rho_list))
ax[-1].legend(handles=legend_elements[type_plot])
xlims = ax[1].get_xlim()
x_values = np.linspace(xlims[0], xlims[1], num=500)
ax[1].plot(x_values, np.power(10, x_values), c='k', ls=':')
ax[1].set_xlim(xlims)
for i, (N_pheno, kbT) in enumerate(type_plot_vs_n_and_kbT_list[type_plot]):
   slope, intercept = N_vs_fit_log[N_pheno]
   ax[1].plot([x for x in x_values if x * slope + intercept > 0], [x for x in slope*x_values + intercept if x > 0], color=color_list[i], lw=0.5, alpha=0.7) 
ax[-1].axis('off')
for i in  range(5):
   ax[i].annotate('ABCDEF'[i], xy=(0.06, 1.1), xycoords='axes fraction', fontsize=12, weight='bold')
f.tight_layout()
f.savefig(filepath + 'plots/DGPmap_analysis.png', bbox_inches='tight', dpi=250)
plt.close('all')

###############################################################################################
print('\n\nplot versatility for deterministic version of synthetic GP map ', flush=True)
###############################################################################################
N_pheno = 100
description_parameters = 'synthetic_normal_' + str(N_pheno)+'_'+ str(L_Dmap) + '_' + str(K_Dmap) 
f, ax = plt.subplots(ncols = 3, figsize=(10, 2.5))
df_versatility = pd.read_csv(filepath + 'data/DGPmap_versatility'+description_parameters+'.csv')
rank_list = [c.split('rank')[-1].strip() for c in df_versatility.columns if c.startswith('mean')]
for i, rank in enumerate(rank_list):
   ax[i].errorbar(np.arange(L_Dmap) + 1,  np.array(df_versatility['mean v  - rank '+str(rank)].tolist()), yerr=np.array(df_versatility['std v - rank '+str(rank)].tolist()), marker='o', alpha=0.8, ms=3, capsize=3)
   ax[i].set_xlabel('position')
   ax[i].set_ylabel('number of neutral mutations')  
   ax[i].plot([0, L_Dmap+1], [K_Dmap-1, K_Dmap-1], c='k')     
   ax[i].plot([0, L_Dmap+1], [0, 0], c='k')    
   ax[i].set_xlim(0.5, L_Dmap + 0.5) 
   assert min(np.arange(L_Dmap) + 1) >= 0.5 and max(np.arange(L_Dmap) + 1) <= L_Dmap 
   ax[i].set_title('phenotype rank ' + str(rank.split(' ')[0]) + ', NC size ' + str(rank.split(' ')[-1]))
f.tight_layout()
f.savefig(filepath + 'plots/DGPmap_versatility'+description_parameters+'.png', dpi=200, bbox_inches='tight')
plt.close('all')
############
f, ax = plt.subplots(nrows = 3, ncols=L_Dmap, figsize=(L_Dmap*3, 9))
for i, rank in enumerate(rank_list):
   pos_vs_hist = {pos: [] for pos in range(L_Dmap)}
   for pos in range(L_Dmap):
      hist = [df_versatility[str(v) + 'neutral prevelance - rank '+str(rank)].tolist()[pos] for v in range(K_Dmap)]
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
f.savefig(filepath + 'plots/DGPmap_versatility'+description_parameters+'_2.png', dpi=200, bbox_inches='tight')
plt.close('all')
############
f, ax = plt.subplots(figsize=(3, 2.2))
df_NCs = pd.read_csv(filepath + 'data/DGPmap_nNCs'+description_parameters+'.csv')
hist_normalised(df_NCs['# NCs'].tolist(), bins=np.arange(0.5, max(df_NCs['# NCs'].tolist())+0.6), ax=ax, color='grey')
ax.set_xlabel('number NCs per phenotype')
ax.set_ylabel('number phenotypes')
print('fraction of phenotypes with #NC > 1', len([x for x in df_NCs['# NCs'].tolist() if x > 1])/len([x for x in df_NCs['# NCs'].tolist() if x >= 1]))
print('have #NC data for ', len([x for x in df_NCs['# NCs'].tolist() if x >= 1]), 'phenotypes')
f.tight_layout()
f.savefig(filepath + 'plots/DGPmap_NC_fragmentation'+description_parameters+'.png', dpi=200, bbox_inches='tight')
plt.close('all')
###############################################################################################
###############################################################################################
description_parameters = 'synthetic_normal_' + str(N_pheno)+'_'+ str(L) + '_' + str(K) 
###############################################################################################
print('\n\nplot data from Gaussian GP map with changes in parameters', flush=True)
###############################################################################################
max_entropy = 0
f, ax = plt.subplots(nrows = 2, ncols=6, figsize=(16, 5))
for columnindex, type_plot in enumerate(['vary_N_pheno', 'vary_T']):
   color_list = type_plot_vs_color_list[type_plot]
   if type_plot == 'vary_T':
      axins = inset_axes(ax[columnindex, 2], width="30%", height="30%", loc="lower left", bbox_to_anchor=(0.6, 0.08, 1, 1), bbox_transform=ax[columnindex, 2].transAxes)
   for i, (N_pheno, kbT) in enumerate(type_plot_vs_n_and_kbT_list[type_plot]):
      description_parameters_kbT = 'synthetic_normal_' + str(N_pheno)+'_'+ str(L) + '_' + str(K) + '_kbT' + str(kbT)
      NDGPmapdata_filename, NDGrobustness_filename, NDGevolvability_filename = filepath + 'data/NDGPmapproperties_'+description_parameters_kbT+'.csv', filepath + 'data/NDGrobustness_'+description_parameters_kbT+'.npy', filepath + 'data/NDGevolvability_'+description_parameters_kbT+'.npy'
      try:
         df_NDGPmap = pd.read_csv(NDGPmapdata_filename)
      except IOError:
         print('not found', NDGPmapdata_filename)
         continue
      ND_N_list = df_NDGPmap['neutral set size'].tolist()
      ND_rho_list = df_NDGPmap['rho'].tolist()
      NDPevolv_list = df_NDGPmap['p_evolv'].tolist()
      max_entropy = max(max(df_NDGPmap['pheno entropy'].tolist()), max_entropy)
      NDGrobustness =  np.load(NDGrobustness_filename)
      NDGevolvability = np.load(NDGevolvability_filename)
      ##
      if len(ND_N_list) > len([N for N in ND_N_list if not np.isnan(N) and N > 0]):
         print(description_parameters_kbT, len([N for N in ND_N_list if not np.isnan(N) and N > 0]), 'out of ', len(ND_N_list), 'phenotypes appear in the map')
      rank_data, N_data_by_rank = np.arange(1, len(ND_N_list) + 1), sorted(ND_N_list, reverse=True)
      ax[columnindex, 0].scatter(np.log10(rank_data), np.log10(np.divide(N_data_by_rank, K**L)), color=color_list[i], s=4, alpha=0.5, lw = 0)
      ax[columnindex, 0].set_ylabel(r"\Large \textbf{{{x}}}".format(x=['A) varying $n_p$', 'B) varying T'][columnindex])+'\n\n'+r'$\log_{10} \tilde f_p$')
      ax[columnindex, 0].set_xlabel(r'$\log_{10}$ freq. rank')
      ##
      vector_sum_list = df_NDGPmap['sum over phenotypic vector'].tolist()
      ax[columnindex, 1].scatter(vector_sum_list, np.log10(np.divide(ND_N_list, K**L)), color=color_list[i], s=4, alpha=0.5, lw = 0)
      ax[columnindex, 1].set_ylabel(r'$\log_{10} \tilde f_p$')
      ax[columnindex, 1].set_xlabel(r'$\sum_i (\vec{v_p})_i$')
      ##
      ax[columnindex, 2].scatter(np.log10(np.divide(ND_N_list, K**L)), np.log10(ND_rho_list), color=color_list[i],  s=4, alpha=0.5, lw = 0)
      ax[columnindex, 2].set_ylabel(r'$\log_{10} \tilde{\rho_p}$')      
      ax[columnindex, 2].set_xlabel(r'$\log_{10} \tilde{f_p}$' )
      if len([i for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] == 0]):
         print('not shown nonzero-freq/norobustness datapoints', len([i for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] == 0]), 'out of', len(ND_rho_list))
      phenos_with_rho_lower_f = [(N/K**L, ND_rho_list[i]) for i, N in enumerate(ND_N_list) if ND_rho_list[i]/(N/K**L) <= 1 ]
      if len(phenos_with_rho_lower_f):
         print(description_parameters_kbT, len([x[1] for x in phenos_with_rho_lower_f]),  'phenotypes with rho <= f have rho max',  max([x[1] for x in phenos_with_rho_lower_f]), 'max f', max([x[0] for x in phenos_with_rho_lower_f]))
      ##
      if type_plot == 'vary_T':
         axins.scatter(np.log10(np.divide(ND_N_list, K**L)), np.log10(ND_rho_list), color=color_list[i], s=2, alpha=0.5, lw = 0)
         axins.set_xticks([-20, -10, 0])
         axins.set_xticklabels(['-20', '-10', '0'], fontsize=7)
         axins.set_yticks([-20, -10, 0])
         axins.set_yticklabels(['-20', '-10', '0'], fontsize=7, rotation='vertical')

      ######
      ax[columnindex, 4].scatter(ND_rho_list, np.array(NDPevolv_list)/(N_pheno - 1), s=3, color=color_list[i], alpha=0.5, lw = 0)
      ax[columnindex, 4].set_ylabel(r'$\tilde{\epsilon_p}$  (normalised)')
      ax[columnindex, 4].set_xlabel(r'$\tilde{\rho_p}$')
      ax[columnindex, 4].set_xlim(-0.05, 1)
      assert np.nanmin(ND_rho_list) >= -0.05 and np.nanmax(ND_rho_list) <= 1
      ######
      ax[columnindex, 3].scatter(NDGrobustness.flatten(), NDGevolvability.flatten(), s=0.5, alpha=0.8, color=color_list[i])
      max_gevolv = max(NDGevolvability.flatten())      
      ax[columnindex, 3].set_ylabel(r'$\tilde{\epsilon_g}$')
      ax[columnindex, 3].set_xlabel(r'$\tilde{\rho_g}$')
      for grho, gev in zip(NDGrobustness.flatten(), NDGevolvability.flatten()):
         assert gev <= (1-grho)* (K-1)*L + 0.0001
   if type_plot == 'vary_T':
       try:
          df_DGPmap = pd.read_csv(filepath + 'data/GPmapproperties_'+ 'synthetic_normal_' + str(N_pheno)+'_'+ str(L) + '_' + str(K) +'.csv')
          D_f_list, D_rho_list = zip(*[(N/K**L, rho) for N, rho in zip(df_DGPmap['neutral set size'].tolist(), df_DGPmap['rho'].tolist()) if not np.isnan(N) and N > 0])
          print('D GP map', len(D_f_list), 'instead of ', len(df_NDGPmap['neutral set size'].tolist()), 'phenotypes')
          ax[columnindex, 0].scatter(np.log10(np.arange(1, 1+ len(D_f_list))), np.log10(sorted(D_f_list, reverse=True)), color='grey', s=4, alpha=0.5, lw = 0)
          ax[columnindex, 2].scatter(np.log10([f for i, f in enumerate(D_f_list) if D_rho_list[i] > 0]), np.log10([rho for i, rho in enumerate(D_rho_list) if D_rho_list[i] > 0]), color='grey', s=4, alpha=0.5, lw = 0)
          slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(D_f_list), D_rho_list)
          x_values = np.linspace(min(xlims[0], ylims[0]), max(xlims[1], ylims[1]), num=500)
          ax[columnindex, 2].plot([x for x in x_values if x*slope + intercept > 0], np.log10([x*slope + intercept for x in x_values if x*slope + intercept > 0]), color='gray')
          ax[columnindex, 4].scatter(df_DGPmap['rho'].tolist(), np.divide(df_DGPmap['p_evolv'].tolist(), N_pheno), color='grey', s=4, alpha=0.5, lw = 0)
       except IOError:
          pass
   ax[columnindex, -1].legend(handles=legend_elements_with_D[type_plot])
   lowest_value_to_plot = -6
   xlims, ylims = ax[columnindex, 2].get_xlim(), ax[columnindex, 2].get_ylim()
   if min([xlims[0], ylims[0]]) < lowest_value_to_plot:
      ax[columnindex, 2].set_xlabel(r'$\log_{10} \tilde{f_p}$' + '\n'+r'($\tilde{f_p} < 10^{-6}$ shown in inset)')
   if type_plot == 'vary_T':
      x_values_inset = [min([xlims[0], ylims[0]]), max(xlims[1], ylims[1])]
      axins.plot(x_values_inset, x_values_inset, c='k', lw=0.5, zorder=-3)
   x_values = [max(min([xlims[0], ylims[0]]), lowest_value_to_plot), max(xlims[1], ylims[1])]
   ax[columnindex, 2].set_xlim(min(x_values), max(x_values))
   assert min(x_values) <= max(min(xlims), lowest_value_to_plot) and max(x_values) >= max(xlims) and min(x_values) <= max(lowest_value_to_plot, min(ylims)) and max(x_values) >= max(ylims)
   ax[columnindex, 2].set_ylim(min(x_values), max(x_values))
   ax[columnindex, 2].plot(x_values, x_values, c='k')
   

for i in range(2):
   ax[i, -1].axis('off')
   ax[i, 3].plot(np.array([0, 1]), (1-np.array([0, 1]))* (K-1)*L, color='grey', ls=':', lw=0.7)

for row in range(2):
   for column in range(5):
      ax[row, column].annotate(r"\textbf{{{x}}}".format(x='ABCDE'[row] + '12345'[column]), xy=(0.06, 1.1), xycoords='axes fraction', fontsize=11, weight='bold')
f.tight_layout()
f.savefig(filepath + 'plots/'+'full'+'NDGPmap_without_DGPmap_vary_T_and_n'+ '.png', bbox_inches='tight', dpi=250)
plt.close('all')
###############################################################################################
print('\n\nSappington robustness analysis', flush=True)
###############################################################################################
f8, ax8 = plt.subplots(ncols=4, nrows=2, figsize=(9, 5), width_ratios=[1, 1, 1, 0.5])
type_plot_vs_minf = {'vary_N_pheno': 1/K**L, 'vary_T': 1/K**L}
for columnindex, type_plot in enumerate(['vary_N_pheno', 'vary_T']):
   color_list = type_plot_vs_color_list[type_plot]
   for i, (N_pheno, kbT) in enumerate(type_plot_vs_n_and_kbT_list[type_plot]):
      description_parameters_kbT = 'synthetic_normal_' + str(N_pheno)+'_'+ str(L) + '_' + str(K) + '_kbT' + str(kbT)
      NDGPmapdata_filename = filepath + 'data/NDGPmapproperties_'+description_parameters_kbT+'.csv'
      try:
         df_NDGPmap = pd.read_csv(NDGPmapdata_filename)
         ND_N_list = df_NDGPmap['neutral set size'].tolist()
         ND_rho_list = df_NDGPmap['rho'].tolist()
         pheno_list = df_NDGPmap['phenotype'].tolist()
         ND_entropy_list = df_NDGPmap['pheno entropy'].tolist()
      except IOError:
         print('not found', NDGPmapdata_filename, filepath + 'data/GPmapproperties_'+description_parameters+'.csv')
         continue
      #####
      ax8[columnindex, 0].set_xlabel(r'$\log_{10} \tilde{f_p}$')
      ax8[columnindex, 0].set_ylabel(r'$\log_{10} \tilde{\rho_p}$') 
      rho, f, entropy = zip(*[(ND_rho_list[i], N/K**L, ND_entropy_list[i]) for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] > 0])
      if len([i for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] == 0]):
         print('not shown nonzero-freq/norobustness datapoints', len([i for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] == 0]), 'out of', len(ND_entropy_list))
      ax8[columnindex, 0].scatter(np.log10(f), np.log10(rho), color=color_list[i], s=3, alpha=0.7)
      for x, y in zip(f, rho):
         if x < K**(1-L):
            assert y < x*K**(L-1)/L
         else:
            assert y < 1 + np.log(x)/(L * np.log(K))
      #####
      predicted_robustness = [K**L*f*ent*np.exp(-1*ent)/(L * np.log(K)) for f, ent in zip(f, entropy)]
      ax8[columnindex, 1].set_xlabel(r'$\tilde{\rho_p}$')
      ax8[columnindex, 1].set_ylabel(r'predicted $\tilde{\rho_p}$' + '\nfrom entropy and freq.')    
      ax8[columnindex, 1].scatter(rho, predicted_robustness, color=color_list[i], s=3, alpha=0.7)  
      ax8[columnindex, 1].plot([0, 1], [0,1], c='k', zorder = -2, lw= 0.5)
      if len([i for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] == 0]):
         print('not shown nonzero-freq/norobustness datapoints', len([i for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] == 0]), 'out of', len(ND_rho_list))
      ####
      ax8[columnindex, 2].set_xlabel(r'phenotype entropy')
      ax8[columnindex, 2].set_ylabel('number phenotypes\n(normalised)')   
      hist_normalised(ND_entropy_list, bins=np.linspace(0, max_entropy), ax=ax8[columnindex, 2], color=color_list[i])
      ###
      type_plot_vs_minf[type_plot] = min(type_plot_vs_minf[type_plot], min(f))


   ax8[columnindex, -1].legend(handles=legend_elements[type_plot])
   ax8[columnindex, -1].axis('off')
for columnindex, type_plot in enumerate(['vary_N_pheno', 'vary_T']):
   xlims = ax8[0, columnindex].get_xlim()
   logx_values = np.linspace(np.log10(type_plot_vs_minf[type_plot] * 0.7), 0, num=10**4)
   x_values = np.power(10, logx_values)
   upper_bound1 = [x*K**(L-1)/L if x < K**(1-L) else np.nan for x in x_values]
   upper_bound2 = [np.nan if x < K**(1-L) else 1 + np.log(x)/(L * np.log(K)) for x in x_values]
   ax8[columnindex, 0].plot(np.log10(x_values), np.log10(upper_bound1), c='grey', zorder=-1, lw=0.9)
   ax8[columnindex, 0].plot(np.log10(x_values), np.log10(upper_bound2), c='grey', zorder=-1, lw=0.9)
for row in range(2):
   for column in range(3):
      ax8[row, column].annotate(r"\textbf{{{x}}}".format(x='ABCD'[row] + '1234'[column]), xy=(0.06, 1.1), xycoords='axes fraction', fontsize=11, weight='bold')

f8.tight_layout(rect=[0,0,1, 0.8])
f8.savefig(filepath + 'plots/'+'full'+'Mohanty_analysis'+description_parameters+'.png', bbox_inches='tight', dpi=250)

plt.close('all')

###############################################################################################
print('\n\nevolvability - normalised vs unnormalised', flush=True)
###############################################################################################
f2, ax2 = plt.subplots(ncols = 2, figsize=(6, 3.7)) 
for columnindex, type_plot in enumerate(['vary_N_pheno', 'vary_T']):
   color_list = type_plot_vs_color_list[type_plot]
   for i, (N_pheno, kbT) in enumerate(type_plot_vs_n_and_kbT_list[type_plot]):
      description_parameters_kbT = 'synthetic_normal_' + str(N_pheno)+'_'+ str(L) + '_' + str(K) + '_kbT' + str(kbT)
      NDGPmapdata_filename = filepath + 'data/NDGPmapproperties_'+description_parameters_kbT+'.csv'
      try:
         df_NDGPmap = pd.read_csv(NDGPmapdata_filename)
      except IOError:
         print('not found', NDGPmapdata_filename)
         continue
      ND_rho_list = df_NDGPmap['rho'].tolist()
      NDPevolv_list = df_NDGPmap['p_evolv'].tolist()
      #
      ax2[columnindex].scatter(ND_rho_list, NDPevolv_list, s=3, color=color_list[i], alpha=0.5, lw = 0)
      ax2[columnindex].set_ylabel(r'$\tilde{\epsilon_p}$')
      ax2[columnindex].set_xlabel(r'$\tilde{\rho_p}$')
      assert -0.05 < min(ND_rho_list) and max(ND_rho_list) <= 1
      xlims = ax2[columnindex].get_xlim()
      ax2[columnindex].set_xlim(-0.05, xlims[1])
      #if columnindex == 0:
      #   ax2[columnindex].set_yscale('log')
      ax2[columnindex].set_title(['varying $n_p$', 'varying T'][columnindex])
   ax2[columnindex].legend(handles=legend_elements[type_plot], loc='lower center', bbox_to_anchor=(0.5, 1.15), ncols=2)
f2.tight_layout()
f2.savefig(filepath + 'plots/'+'full'+'NDGPmap_synthetic_evolv_unnormalised'+ '.png', bbox_inches='tight', dpi=250)
plt.close('all')
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
print('\n\nplot data from synthetic GP maps with different prob distributions -- what is the Boltzmann distribution like', flush=True)
###############################################################################################
###############################################################################################
print('\n\nplot data from synthetic GP maps with different alphabet size', flush=True)
###############################################################################################
f, ax = plt.subplots(nrows = 2, ncols=5, figsize=(12, 4.5), width_ratios = [1.2,] + [1,] * 4)
cmap = plt.get_cmap('cool')
N_pheno = 100
kbT_values_all = [0, 0.1, 0.5, 1, 2.5]
color_list = ['grey',]+[cmap(i) for i in np.linspace(0, 1, len(kbT_values_all))]
for columnindex, K_Dmap in enumerate([2, 4]):
   for i, kbT in enumerate(kbT_values_all):
      L_Dmap = {2: 16, 4: 8}[K_Dmap]
      if kbT > 0:
         description_parameters_kbT = 'synthetic_normal'+'_' + str(N_pheno)+'_'+ str(L_Dmap) + '_' + str(K_Dmap) + '_kbT' + str(kbT)
         NDGPmapdata_filename, NDGrobustness_filename, NDGevolvability_filename = filepath + 'data/NDGPmapproperties_'+description_parameters_kbT+'.csv', filepath + 'data/NDGrobustness_'+description_parameters_kbT+'.npy', filepath + 'data/NDGevolvability_'+description_parameters_kbT+'.npy'
      else:
        description_parameters_kbT = 'synthetic_normal'+'_'+ str(N_pheno)+'_'+ str(L_Dmap) + '_' + str(K_Dmap)
        NDGPmapdata_filename, NDGrobustness_filename, NDGevolvability_filename = filepath + 'data/GPmapproperties_'+description_parameters_kbT+'.csv', filepath + 'data/Grobustness_'+description_parameters_kbT+'.npy', filepath + 'data/Gevolvability_'+description_parameters_kbT+'.npy'

      try:
         df_NDGPmap = pd.read_csv(NDGPmapdata_filename)
         NDGrobustness =  np.load(NDGrobustness_filename)
         NDGevolvability = np.load(NDGevolvability_filename)
         print(NDGrobustness_filename, NDGevolvability_filename)
      except IOError:
         print('not found', NDGPmapdata_filename)
         continue
      ND_N_list = df_NDGPmap['neutral set size'].tolist()
      ND_rho_list = df_NDGPmap['rho'].tolist()
      NDPevolv_list = df_NDGPmap['p_evolv'].tolist()
      ##
      ND_N_list_nonan = [N for N in ND_N_list if N > 0 and not np.isnan(N)]
      if  kbT > 0 and len(ND_N_list) > len(ND_N_list_nonan):
         print(description_parameters_kbT, len(ND_N_list_nonan), 'out of ', len(ND_N_list), 'phenotypes appear in the map')
      rank_data, N_data_by_rank = np.arange(1, len(ND_N_list_nonan) + 1), sorted(ND_N_list_nonan, reverse=True)
      ax[columnindex, 0].scatter(np.log10(rank_data), np.log10(np.divide(N_data_by_rank, K_Dmap**L_Dmap)), color=color_list[i], s=4, alpha=0.5, lw = 0)
      ax[columnindex, 0].set_ylabel(r'$\bf{{{0}}}$'.format('K='+str(K_Dmap)+', L='+str(L_Dmap))+'\n\n'+r'$\log_{10} \tilde f_p$')
      ax[columnindex, 0].set_xlabel(r'$\log_{10}$ freq. rank')
      ##
      rho, freq = zip(*[(ND_rho_list[i], N/K_Dmap**L_Dmap) for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] > 0])
      ax[columnindex, 1].scatter(np.log10(freq), np.log10(rho), color=color_list[i], s=2, alpha=0.5, lw = 0)
      ax[columnindex, 1].set_ylabel(r'$\log_{10} \tilde{\rho_p}$')
      ax[columnindex, 1].set_xlabel(r'$\log_{10} \tilde{f_p}$' )  
      if len([i for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] == 0]):
         print('not shown nonzero-freq/norobustness datapoints', len([i for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] == 0]), 'out of', len(ND_rho_list))   
      phenos_with_rho_lower_f = [(N/K_Dmap**L_Dmap, ND_rho_list[i]) for i, N in enumerate(ND_N_list) if ND_rho_list[i]/(N/K_Dmap**L_Dmap) <= 1 ]
      if len(phenos_with_rho_lower_f):
         print(description_parameters_kbT, len([x[1] for x in phenos_with_rho_lower_f]),  'phenotypes with rho < f')
      ######
      ax[columnindex, 3].scatter(ND_rho_list, np.array(NDPevolv_list)/(N_pheno - 1), s=3, color=color_list[i], alpha=0.5, lw = 0)
      ax[columnindex, 3].set_ylabel(r'$\tilde{\epsilon_p}$  (normalised)')
      ax[columnindex, 3].set_xlabel(r'$\tilde{\rho_p}$')
      ax[columnindex, 3].set_xlim(-0.05, 1)
      assert -0.05 < np.nanmin(ND_rho_list) and np.nanmax(ND_rho_list) <= 1
      ######
      ax[columnindex, 2].scatter(NDGrobustness.flatten(), NDGevolvability.flatten(), s=0.5, alpha=0.4, color=color_list[i])
      max_gevolv = max(NDGevolvability.flatten())
      ax[columnindex, 2].set_ylabel(r'$\tilde{\epsilon_g}$')
      ax[columnindex, 2].set_xlabel(r'$\tilde{\rho_g}$')
      for i, (grho, gev) in enumerate(zip(NDGrobustness.flatten(), NDGevolvability.flatten())):
         assert (gev <= (1-grho)* (K_Dmap-1)*L_Dmap + 0.0001) or (np.isnan(grho) and kbT == 0)
      print('------')
   xlims = ax[columnindex, 1].get_xlim()
   ylims = ax[columnindex, 1].get_ylim()
   custom_lines = [Line2D([0], [0], mfc='grey', ls='', marker='o', label=r'deterministic', mew=0, ms=5),] + [Line2D([0], [0], mfc=color_list[i], ls='', marker='o', label='$T$='+str(kbT), mew=0, ms=5) for i, kbT in enumerate(kbT_values_all) if kbT > 0]
   ax[columnindex, -1].legend(handles=custom_lines)
   x_values = np.linspace(min(xlims[0], ylims[0]), max(xlims[1], ylims[1]), num=500)
   ax[columnindex, 1].set_xlim(min(xlims[0], ylims[0]), max(xlims[1], ylims[1]))
   ax[columnindex, 1].set_ylim(min(xlims[0], ylims[0]), max(xlims[1], ylims[1]))
   assert xlims[0] <= xlims[1] and ylims[0] <= ylims[1]
   ax[columnindex, 1].plot(x_values, x_values, c='k', lw=0.5, zorder=-2)
   ####
   ax[columnindex, 2].plot(np.array([0, 1]), (1-np.array([0, 1]))* (K_Dmap-1)*L_Dmap, color='grey', ls=':', lw=0.7)
for row in range(2):
   ax[row, -1].axis('off')
   for column in range(4):
      ax[row, column].annotate(r"\textbf{{{x}}}".format(x='ABCDE'[row] + '12345'[column]), xy=(0.06, 1.1), xycoords='axes fraction', fontsize=11, weight='bold')

f.tight_layout()
f.savefig(filepath + 'plots/'+'vary_K'+'NDGPmap_without_DGPmap_vary_T_and_n'+ '.png', bbox_inches='tight', dpi=250)
plt.close('all')
###############################################################################################
print('\n\nplot data from synthetic GP maps with different prob distributions', flush=True)
###############################################################################################
f, ax = plt.subplots(nrows = 4, ncols=5, figsize=(12, 8), width_ratios = [1.2,] + [1,] * 4)
cmap = plt.get_cmap('cool')
N_pheno = 100
color_list = ['grey',]+[cmap(i) for i in np.linspace(0, 1, len(kbT_values_all))]
for columnindex, type_plot in enumerate(['exponential', 'lognormal', 'uniform', 'normal']):
   for i, kbT in enumerate(kbT_values_all):
      if kbT > 0:
         description_parameters_kbT = 'synthetic_'+type_plot+'_' + str(N_pheno)+'_'+ str(L) + '_' + str(K) + '_kbT' + str(kbT)
         NDGPmapdata_filename, NDGrobustness_filename, NDGevolvability_filename = filepath + 'data/NDGPmapproperties_'+description_parameters_kbT+'.csv', filepath + 'data/NDGrobustness_'+description_parameters_kbT+'.npy', filepath + 'data/NDGevolvability_'+description_parameters_kbT+'.npy'
      else:
        description_parameters_kbT = 'synthetic_'+type_plot+'_'+ str(N_pheno)+'_'+ str(L) + '_' + str(K)
        NDGPmapdata_filename, NDGrobustness_filename, NDGevolvability_filename = filepath + 'data/GPmapproperties_'+description_parameters_kbT+'.csv', filepath + 'data/Grobustness_'+description_parameters_kbT+'.npy', filepath + 'data/Gevolvability_'+description_parameters_kbT+'.npy'

      try:
         df_NDGPmap = pd.read_csv(NDGPmapdata_filename)
         NDGrobustness =  np.load(NDGrobustness_filename)
         NDGevolvability = np.load(NDGevolvability_filename)
      except IOError:
         print('not found', NDGPmapdata_filename)
         continue
      ND_N_list = df_NDGPmap['neutral set size'].tolist()
      ND_rho_list = df_NDGPmap['rho'].tolist()
      NDPevolv_list = df_NDGPmap['p_evolv'].tolist()
      
      ##
      ND_N_list_nonan = [N for N in ND_N_list if N > 0 and not np.isnan(N)]
      if  kbT > 0 and len(ND_N_list) > len(ND_N_list_nonan):
         print(description_parameters_kbT, len(ND_N_list_nonan), 'out of ', len(ND_N_list), 'phenotypes appear in the map')
      rank_data, N_data_by_rank = np.arange(1, len(ND_N_list_nonan) + 1), sorted(ND_N_list_nonan, reverse=True)
      ax[columnindex, 0].scatter(np.log10(rank_data), np.log10(np.divide(N_data_by_rank, K**L)), color=color_list[i], s=4, alpha=0.5, lw = 0)
      ax[columnindex, 0].set_ylabel(r'$\bf{{{0}}}$'.format(type_plot)+'\n\n'+r'$\log_{10} \tilde f_p$')
      ax[columnindex, 0].set_xlabel(r'$\log_{10}$ freq. rank')
      ##
      rho, freq = zip(*[(ND_rho_list[i], N/K**L) for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] > 0])
      ax[columnindex, 1].scatter(np.log10(freq), np.log10(rho), color=color_list[i], s=2, alpha=0.5, lw = 0)
      ax[columnindex, 1].set_ylabel(r'$\log_{10} \tilde{\rho_p}$')
      ax[columnindex, 1].set_xlabel(r'$\log_{10} \tilde{f_p}$' )  
      phenos_with_rho_lower_f = [(N/K**L, ND_rho_list[i]) for i, N in enumerate(ND_N_list) if ND_rho_list[i]/(N/K**L) <= 1 ]
      if len(phenos_with_rho_lower_f):
         print(description_parameters_kbT, len([x[1] for x in phenos_with_rho_lower_f]),  'phenotypes with rho <= f have rho max out of', len([N for N in ND_N_list_nonan if N > 0]),  max([x[1] for x in phenos_with_rho_lower_f]), 'max f', max([x[0] for x in phenos_with_rho_lower_f]))
      if len([i for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] == 0]):
         print('not shown nonzero-freq/norobustness datapoints', len([i for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] == 0]), 'out of', len(ND_rho_list))  
      print(description_parameters_kbT, 'mean ratio np.log10(rho/f)', np.mean(np.log10([ND_rho_list[i]/(N/K**L) for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] > 0])))
 
      ######
      ax[columnindex, 3].scatter(ND_rho_list, np.array(NDPevolv_list)/(N_pheno - 1), s=3, color=color_list[i], alpha=0.5, lw = 0)
      ax[columnindex, 3].set_ylabel(r'$\tilde{\epsilon_p}$  (normalised)')
      ax[columnindex, 3].set_xlabel(r'$\tilde{\rho_p}$')
      ax[columnindex, 3].set_xlim(-0.05, 1)
      assert -0.05 < np.nanmin(ND_rho_list) and np.nanmax(ND_rho_list) <= 1
      ######
      ax[columnindex, 2].scatter(NDGrobustness.flatten(), NDGevolvability.flatten(), s=0.5, alpha=0.4, color=color_list[i])
      max_gevolv = max(NDGevolvability.flatten())
      ax[columnindex, 2].set_ylabel(r'$\tilde{\epsilon_g}$')
      ax[columnindex, 2].set_xlabel(r'$\tilde{\rho_g}$')
      for i, (grho, gev) in enumerate(zip(NDGrobustness.flatten(), NDGevolvability.flatten())):
         assert (gev <= (1-grho)* (K-1)*L + 0.0001) or (np.isnan(grho) and kbT == 0)
      print('---')
   xlims = ax[columnindex, 1].get_xlim()
   ylims = ax[columnindex, 1].get_ylim()
   custom_lines = [Line2D([0], [0], mfc='grey', ls='', marker='o', label=r'deterministic', mew=0, ms=5),] + [Line2D([0], [0], mfc=color_list[i], ls='', marker='o', label='$T$='+str(kbT), mew=0, ms=5) for i, kbT in enumerate(kbT_values_all) if kbT > 0]
   ax[columnindex, -1].legend(handles=custom_lines)
   x_values = np.linspace( min(xlims[0], ylims[0]), max(xlims[1], ylims[1]), num=500)
   ax[columnindex, 1].set_xlim(min(xlims[0], ylims[0]), max(xlims[1], ylims[1]))
   ax[columnindex, 1].set_ylim(min(xlims[0], ylims[0]), max(xlims[1], ylims[1]))
   assert xlims[0] <= xlims[1] and ylims[0] <= ylims[1]
   ax[columnindex, 1].plot(x_values, x_values, c='k', lw=0.5, zorder=-2)
   ####
for i in range(4):
   ax[i, -1].axis('off')
   ax[i, 2].plot(np.array([0, 1]), (1-np.array([0, 1]))* (K-1)*L, color='grey', ls=':', lw=0.7)

for row in range(4):
   for column in range(4):
      ax[row, column].annotate(r"\textbf{{{x}}}".format(x='ABCDE'[row] + '12345'[column]), xy=(0.06, 1.1), xycoords='axes fraction', fontsize=11, weight='bold')

f.tight_layout()
f.savefig(filepath + 'plots/'+'nonGaussian_version'+'NDGPmap_without_DGPmap_vary_T_and_n'+ '.png', bbox_inches='tight', dpi=250)
plt.close('all')
###############################################################################################
print('\n\nplot peaks from synthetic GP maps with different prob distributions', flush=True)
###############################################################################################
f2, ax2 = plt.subplots(ncols=4, figsize=(12, 3))
f3, ax3 = plt.subplots(ncols=4, figsize=(12, 3))
cmap = plt.get_cmap('cool')
N_pheno = 100
for columnindex, type_plot in enumerate(['exponential', 'lognormal', 'uniform', 'normal']):
   for i, kbT in enumerate(kbT_values_all):
      if kbT == 0:
         continue
      description_parameters_kbT =  'synthetic_'+type_plot+'_' + str(N_pheno)+'_'+ str(L) + '_' + str(K) + '_kbT' + str(kbT)
      list_distances_peaks = np.load(filepath + 'data/list_distances_peaks'+description_parameters_kbT+'.npy') 
      Hamming_dist_range = [H - 0.5 for H in range(L + 2)]
      hist_normalised(list_distances_peaks, bins=Hamming_dist_range, ax=ax2[columnindex], color=color_list[i])
      ax2[columnindex].plot(list(range(L + 1)), [comb(L, H)*(K-1)**H/(float(K**L)) for H in range(L + 1)], c='k')
      ax2[columnindex].set_xlabel('distance between genotypes\nmaximising p and q')
      ax2[columnindex].set_ylabel('frequency\n(normalised)')
      ax2[columnindex].legend(handles=[Line2D([0], [0], mfc=color_list[i], ls='', marker='o', label='$T$='+str(kbT), mew=0, ms=5) for i, kbT in enumerate(kbT_values_all) if kbT > 0], 
                              loc='lower center', bbox_to_anchor=(0.5, 1.12), ncols=2)  
      ax2[columnindex].set_title(type_plot)
      print(description_parameters_kbT, 'mean distance peaks', np.mean(list_distances_peaks))
      #
      df_peaks = pd.read_csv(filepath + 'data/list_peaks'+description_parameters_kbT+'.csv')
      pos_vs_mean_across_peaks = [np.mean([int(p.split('_')[pos]) for p in df_peaks['peak'].tolist() if int(p.split('_')[pos]) >= -0.1]) for pos in range(L)]
      pos_vs_max_abs_vector_element = [np.max([float(p.split('_')[pos]) for p in df_peaks['vector'].tolist()]) for pos in range(L)]
      ax3[columnindex].scatter(pos_vs_mean_across_peaks, pos_vs_max_abs_vector_element, color=color_list[i], s=4)
      ax3[columnindex].set_xlim(0, 1)
      assert 0 <= min(pos_vs_mean_across_peaks) and max(pos_vs_mean_across_peaks) <= 1
      ax3[columnindex].set_xlabel('mean value at\ngenotype position i\nacross all peak genotypes')
      ax3[columnindex].set_ylabel('maximum value at\nvector position i\n'+r'among $\vec{v}_p$ for all phenotypes $p$')
      ax3[columnindex].legend(handles=[Line2D([0], [0], mfc=color_list[i], ls='', marker='o', label='$T$='+str(kbT), mew=0, ms=5) for i, kbT in enumerate(kbT_values_all) if kbT > 0], 
                              loc='lower center', bbox_to_anchor=(0.5, 1.12), ncols=2)  
      ax3[columnindex].set_title(type_plot)
   ####
f2.tight_layout()
f2.savefig(filepath + 'plots/'+'nonGaussian_version'+'list_distances_peaks'+ '.png', bbox_inches='tight', dpi=250)
f3.tight_layout()
f3.savefig(filepath + 'plots/'+'nonGaussian_version'+'list_distances_peaks_analysis'+ '.png', bbox_inches='tight', dpi=250)
plt.close('all')

###############################################################################################
print('\n\nplot data from synthetic GP maps with different functional forms -- what is the Boltzmann distribution like', flush=True)
###############################################################################################
cmap = plt.get_cmap('cool')
N_pheno = 100
f, ax = plt.subplots(ncols=3, figsize=(7, 2.5), width_ratios = [1,1, 0.3])
color_list, distribution_list = ['r', 'b', 'g', 'y'], ['linear', 'inversesquared', 'expsquared', 'Boltzmann-like'] 
for columnindex, type_plot in enumerate(distribution_list):
   kbT_values = [0.01, 0.02, 0.05, 0.1, 0.2,  0.5, 1, 2.5, 5]
   list_median_Pmfe, list_median_ratio_suboptimal, kbT_values_withdata = [], [], []
   for kbT in kbT_values:
      try:
         if type_plot == 'Boltzmann-like':
            description_parameters_kbT = 'synthetic_normal_' + str(N_pheno)+'_'+ str(L) + '_' + str(K) + '_kbT' + str(kbT)
         else:
            description_parameters_kbT = 'synthetic_diffnonlinear_function_'+type_plot+'_normal_' + str(N_pheno)+'_'+ str(L) + '_' + str(K) + '_kbT' + str(kbT)
         mfe_P_array = np.load(filepath + 'data/mfe_P_array_'+description_parameters_kbT+'.npy')      
         first_suboptimal_P_array = np.load(filepath + 'data/first_suboptimal_P_array_'+description_parameters_kbT+'.npy')
         assert np.nanmin(mfe_P_array.flatten()[1:]) > 0 #undefined only exists for genotype (0,0,0); all other have to have some defined probability
         ratio_data = [x/y for x, y in zip(first_suboptimal_P_array.flatten()[1:], mfe_P_array.flatten()[1:]) ]
         mfe_data = mfe_P_array.flatten()[1:]
         list_median_Pmfe.append((float(np.percentile(mfe_data, 25)), float(np.median(mfe_data)), float(np.percentile(mfe_data, 75))))
         list_median_ratio_suboptimal.append((float(np.percentile(ratio_data, 25)), float(np.median(ratio_data)), float(np.percentile(ratio_data, 75))))
         kbT_values_withdata.append(kbT)
      except IOError:
         print('not found', filepath + 'data/mfe_P_array_'+description_parameters_kbT+'.npy')
         continue
   if len(list_median_Pmfe) == 0:
      continue
   lower_q, median, upper_q = zip(*list_median_Pmfe)
   ax[0].errorbar(kbT_values_withdata, median, yerr=(np.array(median) - np.array(lower_q), np.array(upper_q)- np.array(median)), color = color_list[columnindex])
   lower_q2, median2, upper_q2 = zip(*list_median_ratio_suboptimal)
   ax[1].errorbar(kbT_values_withdata, median2, yerr=(np.array(median2) - np.array(lower_q2), np.array(upper_q2)- np.array(median2)), color = color_list[columnindex])

ax[0].set_yscale('log')
ax[0].set_ylabel(r'$P(p_1|g)$ for highest-prob. $p_1$')
ax[1].set_ylabel(r'ratio: $P(p_2|g)/P(p_1|g)$' +'\n' + 'for highest-prob. $p_1$'+'\n' + 'and 2nd-highest-prob. $p_1$')
ax[0].plot(kbT_values, [1/N_pheno,] * len(kbT_values), c='grey', ls=':')
ax[1].plot(kbT_values, [1,] * len(kbT_values), c='grey', ls=':')
for i in range(2):
   ax[i].set_xlabel(r'stochasticity $T$')
   ax[i].set_xscale('log')
custom_lines = [Line2D([0], [0], mfc=color_list[i], ls='', marker='o', label=type_plot, mew=0, ms=5) for i, type_plot in enumerate(distribution_list)]
custom_lines.append(Line2D([0], [0], mfc='grey', ls=':', marker=None, label='null model', mew=0, ms=5, c='grey') )
ax[-1].legend(handles=custom_lines)
ax[-1].axis('off')
f.tight_layout()
f.savefig(filepath + 'plots/non_Boltz_nonlinear_funct_stats_vary_t.png', bbox_inches='tight', dpi=250)
plt.close('all')

###############################################################################################
print('\n\nplot data from synthetic GP maps with different nonlinear functions', flush=True)
###############################################################################################
kbT_values_all = [0, 0.1, 0.5, 1, 2.5]
f, ax = plt.subplots(nrows = 3, ncols=5, figsize=(14, 7), width_ratios = [1.2,] + [1,] * 4)
for columnindex, type_plot in enumerate(['linear', 'inversesquared', 'expsquared']):
   color_list = ['grey',]+[cmap(i) for i in np.linspace(0, 1, 5)]
   for i, kbT in enumerate(kbT_values_all):
      if kbT > 0:
         description_parameters_kbT = 'synthetic_diffnonlinear_function_'+type_plot+'_normal_' + str(N_pheno)+'_'+ str(L) + '_' + str(K) + '_kbT' + str(kbT)
         NDGPmapdata_filename, NDGrobustness_filename, NDGevolvability_filename = filepath + 'data/NDGPmapproperties_'+description_parameters_kbT+'.csv', filepath + 'data/NDGrobustness_'+description_parameters_kbT+'.npy', filepath + 'data/NDGevolvability_'+description_parameters_kbT+'.npy'
      else:
        description_parameters_kbT = 'synthetic_diffnonlinear_function_'+type_plot+'_normal_' + str(N_pheno)+'_'+ str(L) + '_' + str(K) 
        NDGPmapdata_filename, NDGrobustness_filename, NDGevolvability_filename = filepath + 'data/GPmapproperties_'+description_parameters_kbT+'.csv', filepath + 'data/Grobustness_'+description_parameters_kbT+'.npy', filepath + 'data/Gevolvability_'+description_parameters_kbT+'.npy'

      try:
         df_NDGPmap = pd.read_csv(NDGPmapdata_filename)
         NDGrobustness =  np.load(NDGrobustness_filename)
         NDGevolvability = np.load(NDGevolvability_filename)
      except IOError:
         print('not found', NDGPmapdata_filename)
         continue
      ND_N_list = df_NDGPmap['neutral set size'].tolist()
      ND_rho_list = df_NDGPmap['rho'].tolist()
      NDPevolv_list = df_NDGPmap['p_evolv'].tolist()      
      ##
      ND_N_list_nonzero = [N for N in ND_N_list if not np.isnan(N) and N> 0]
      if kbT > 0 and len(ND_N_list) > len(ND_N_list_nonzero):
         print(description_parameters_kbT, len(ND_N_list_nonzero), 'out of ', len(ND_N_list), 'phenotypes appear in the map')
      rank_data, N_data_by_rank = np.arange(1, len(ND_N_list_nonzero) + 1), sorted(ND_N_list_nonzero, reverse=True)
      ax[columnindex, 0].scatter(np.log10(rank_data), np.log10(np.divide(N_data_by_rank, K**L)), color=color_list[i], s=4, alpha=0.5, lw = 0)
      ax[columnindex, 0].set_ylabel(r'$\bf{{{0}}}$'.format(type_plot)+'\n\n'+r'$\log_{10} \tilde f_p$')
      ax[columnindex, 0].set_xlabel(r'$\log_{10}$ freq. rank')
      if type_plot == 'linear' and kbT > 0:
         print('linear map, ratio maxf to minf', max(N_data_by_rank)/min(N_data_by_rank), 'zero-freq', len(ND_N_list) - len(ND_N_list_nonzero))
      ##
      rho, freq = zip(*[(ND_rho_list[i], N/K**L) for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] > 0])
      ax[columnindex, 1].scatter(np.log10(freq), np.log10(rho), color=color_list[i], s=4, alpha=0.5, lw = 0)
      ax[columnindex, 1].set_ylabel(r'$\log_{10} \tilde{\rho_p}$')
      ax[columnindex, 1].set_xlabel(r'$\log_{10} \tilde{f_p}$')
      print(description_parameters_kbT, 'mean ratio np.log10(rho/f)', np.mean(np.log10([ND_rho_list[i]/(N/K**L) for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] > 0])))
      if len([i for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] == 0]):
         print('not shown nonzero-freq/norobustness datapoints', len([i for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] == 0]), 'out of', len(ND_rho_list))
      phenos_with_rho_lower_f = [(N/K**L, ND_rho_list[i]) for i, N in enumerate(ND_N_list) if ND_rho_list[i]/(N/K**L) <= 1 ]
      if len(phenos_with_rho_lower_f):
         print(description_parameters_kbT, len([x[1] for x in phenos_with_rho_lower_f]), 'phenotypes with rho <= f have rho max',  max([x[1] for x in phenos_with_rho_lower_f]), 'max f', max([x[0] for x in phenos_with_rho_lower_f]))
      ######
      ax[columnindex, 3].scatter(ND_rho_list, np.array(NDPevolv_list)/(len(NDPevolv_list) - 1), s=3, color=color_list[i], alpha=0.5, lw = 0)
      ax[columnindex, 3].set_ylabel(r'$\tilde{\epsilon_p}$  (normalised)')
      ax[columnindex, 3].set_xlabel(r'$\tilde{\rho_p}$')
      ax[columnindex, 3].set_xlim(-0.05, 1)
      assert np.nanmin(ND_rho_list) >= -0.05 and np.nanmax(ND_rho_list) <= 1
      if kbT > 0 and type_plot == 'linear':
         print('min evolv in ', description_parameters_kbT, '1-',  1-min(np.array(NDPevolv_list)/(len(NDPevolv_list) - 1)))
      ######
      ax[columnindex, 2].scatter(NDGrobustness.flatten(), NDGevolvability.flatten(), s=0.5, alpha=0.8, color=color_list[i])
      max_gevolv = max(NDGevolvability.flatten())
      ax[columnindex, 2].set_ylabel(r'$\tilde{\epsilon_g}$')
      ax[columnindex, 2].set_xlabel(r'$\tilde{\rho_g}$')  
      for grho, gev in zip(NDGrobustness.flatten(), NDGevolvability.flatten()):
         if not ((gev <= (1-grho)* (K-1)*L + 0.0001) or (np.isnan(grho) and kbT == 0)):
            print(description_parameters_kbT, kbT, grho, gev)
         assert (gev <= (1-grho)* (K-1)*L + 0.0001) or (np.isnan(grho) and kbT == 0)
      if type_plot == 'linear' and kbT > 0:
         print('linear map, min gev, max grho', min(NDGevolvability.flatten()[1:]), max(NDGrobustness.flatten()[1:]), 'exception all-0 geno with', NDGevolvability.flatten()[0], NDGrobustness.flatten()[0])
      print('------')    
   custom_lines = [Line2D([0], [0], mfc='grey', ls='', marker='o', label=r'deterministic', mew=0, ms=5),] + [Line2D([0], [0], mfc=color_list[i], ls='', marker='o', label='$T$='+str(kbT), mew=0, ms=5) for i, kbT in enumerate(kbT_values_all) if kbT > 0]
   ax[columnindex, -1].legend(handles=custom_lines)
   xlims = ax[columnindex, 1].get_xlim()
   ylims = ax[columnindex, 1].get_ylim()
   x_values = np.linspace(min(xlims[0], ylims[0]), max(xlims[1], ylims[1]), num=500)
   ax[columnindex, 1].set_xlim(min(xlims[0], ylims[0]), max(xlims[1], ylims[1]))
   ax[columnindex, 1].set_ylim(min(xlims[0], ylims[0]), max(xlims[1], ylims[1]))
   assert xlims[0] <= xlims[1] and ylims[0] <= ylims[1]
   ax[columnindex, 1].plot(x_values, x_values, c='k', lw=0.5, zorder=-2)
for i in range(3):
   ax[i, -1].axis('off')
   ax[i, 2].plot(np.array([0, 1]), (1-np.array([0, 1]))* (K-1)*L, color='grey', ls=':', lw=0.7)
for row in range(3):
   for column in range(4):
      ax[row, column].annotate(r"\textbf{{{x}}}".format(x='ABCDE'[row] + '12345'[column]), xy=(0.06, 1.1), xycoords='axes fraction', fontsize=11, weight='bold')

f.tight_layout()
f.savefig(filepath + 'plots/'+'non_Boltz_nonlinear_funct'+'NDGPmap_without_DGPmap_vary_T_and_n'+ '.png', bbox_inches='tight', dpi=250)
plt.close('all')

