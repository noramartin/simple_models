#!/usr/bin/env python3

import sys
print(sys.version)
import numpy as np
from functions import synthetic_model_functions as synthetic
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import isfile
from scipy import stats
from math import comb
import pandas as pd
from matplotlib.lines import Line2D
import seaborn as sns
import parameters as param
plt.rcParams["text.usetex"] = True
from scipy.stats import pearsonr
from functions.general_functions import *


def set_up_analysis(kbt_list, description_parameters):
	cmap = plt.get_cmap('cool')
	color_list = [cmap(i) for i in np.linspace(0, 0.9, len(kbt_list))]
	if description_parameters != 'polyomino' and not description_parameters.startswith('synthetic'):
		K, L = type_map_vs_K[description_parameters], type_map_vs_L[description_parameters]
	else:
		K, L = np.nan, np.nan
	if description_parameters.startswith('polyomino'):
		label_list = ['$S_{%s, %s}$' %(kbT.split('_')[0], kbT.split('_')[1]) for i, kbT in enumerate(kbt_list)]
	elif description_parameters.startswith('RNA'):
		label_list = ['$T$='+str(kbT)+ r'$^{o}$C' for i, kbT in enumerate(kbt_list)]
	else:
		label_list = ['$k_bT$='+str(kbT) for i, kbT in enumerate(kbt_list)]
	custom_lines = [Line2D([0], [0], mfc=color_list[i], ls='', marker='o', label=label_list[i], mew=0, ms=5) for i, kbT in enumerate(kbt_list)]
	return custom_lines, K, L, color_list, label_list




filepath = './'
polyomino_params = '_'+ '_'.join(param.polyomino_filename.split('_')[2:])
type_map_vs_kbT_list = {'RNA12': [30, 37, 45], 'polyomino': ['2_8', '3_3'], 'HP16_2Li': [0.001, 0.25, 0.5, 1]}
type_map_vs_K, type_map_vs_L = {'RNA12': 4, 'HP16_2Li': 2}, {'RNA12': 12, 'HP16_2Li': 16}
description_param_to_title = {'RNA12': 'RNA', 'polyomino': 'Polyomino', 'synthetic_normal_100_15_2': 'synthetic model', 'HP16_2Li': r'lattice protein'}
list_to_plot = ['RNA12', 'polyomino', 'HP16_2Li']

###############################################################################################
print('\n\nplot main ND GP map figure', flush=True)
###############################################################################################
model_vs_exceptions_g_corr = {}
f, ax = plt.subplots(ncols = 5, nrows=3, figsize=(9, 5), width_ratios=[1, 1, 1, 1, 0.5])
type_map_vs_minmax_freq, type_map_vs_minmax_rho_freq_ratio = {t : [10**3, 1] for t in type_map_vs_kbT_list}, {t : [1, 1] for t in type_map_vs_kbT_list}
for columnindex, description_parameters in enumerate(list_to_plot):
	custom_lines, K, L, color_list, label_list = set_up_analysis(type_map_vs_kbT_list[description_parameters], description_parameters)
	max_gevolv = 0
	for i, kbT in enumerate(type_map_vs_kbT_list[description_parameters]):
		if description_parameters == 'polyomino':
			description_parameters_kbT = description_parameters + str(kbT) + polyomino_params
			K, L = int(kbT.split('_')[1]), int(kbT.split('_')[0]) * 4
		else:
			description_parameters_kbT = description_parameters +'_kbT'+str(kbT)
		NDGPmapdata_filename, NDGrobustness_filename, NDGevolvability_filename = filepath + 'data/NDGPmapproperties_'+description_parameters_kbT+'.csv', filepath + 'data/NDGrobustness_'+description_parameters_kbT+'.npy', filepath + 'data/NDGevolvability_'+description_parameters_kbT+'.npy'
		try:
			df_NDGPmap = pd.read_csv(NDGPmapdata_filename)
		except IOError:
			print('not found', NDGPmapdata_filename)
			continue
		ND_N_list = df_NDGPmap['neutral set size'].tolist()
		ND_rho_list = df_NDGPmap['rho'].tolist()
		NDPevolv_list = df_NDGPmap['p_evolv'].tolist()
		NDGrobustness =  np.load(NDGrobustness_filename)
		NDGevolvability = np.load(NDGevolvability_filename)
		####

		
		##
		ND_N_list_nonzero = [N for N in ND_N_list if not np.isnan(N) and N > 0]
		N_pheno = len(ND_N_list_nonzero)
		ax[columnindex, 0].scatter(np.log10(np.arange(1, len(ND_N_list_nonzero) + 1)), np.log10(sorted(np.divide(ND_N_list_nonzero, K**L), reverse=True)), color=color_list[i], s=3, alpha=0.5, lw = 0)
		ax[columnindex, 0].set_ylabel(r'\Large \textbf{{{x}}}'.format(x='ABCD'[columnindex]+') '+description_param_to_title[description_parameters])+'\n\n'+r'$\log_{10} \tilde{f_p}$') #+r'$\log_{10} \tilde{f_p}$'+ '\n'
		ax[columnindex, 0].set_xlabel(r'$\log_{10}$ freq. rank')
		ax[columnindex, 0].set_xticks([0, 1, 2])
		###
		type_map_vs_minmax_freq[description_parameters][0] = min([N for N in ND_N_list if N > 0] + [type_map_vs_minmax_freq[description_parameters][0],])
		type_map_vs_minmax_freq[description_parameters][1] = max([N for N in ND_N_list if N > 0] + [type_map_vs_minmax_freq[description_parameters][1],])
		type_map_vs_minmax_rho_freq_ratio[description_parameters][0] = min([type_map_vs_minmax_rho_freq_ratio[description_parameters][0],] + [ND_rho_list[i]/(N/K**L) for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] > 0])
		type_map_vs_minmax_rho_freq_ratio[description_parameters][1] = max([type_map_vs_minmax_rho_freq_ratio[description_parameters][1],] + [ND_rho_list[i]/(N/K**L) for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] > 0])

		##
		ND_N_list_nonzero, ND_rho_list_nonzero = [N for i, N in enumerate(ND_N_list) if not np.isnan(N) and N > 0 and ND_rho_list[i] > 0], [N for i, N in enumerate(ND_rho_list) if not np.isnan(ND_N_list[i]) and N > 0 and ND_N_list[i] > 0]
		ax[columnindex, 1].scatter(np.log10(np.divide(ND_N_list_nonzero, K**L)), np.log10(ND_rho_list_nonzero), color=color_list[i], s=3, alpha=0.5, lw = 0)
		ax[columnindex, 1].set_ylabel(r'$\log_{10} \tilde{\rho_p}$')#r'$\log_{10} \tilde{\rho_p}$'+ '\n'+
		ax[columnindex, 1].set_xlabel(r'$\log_{10} \tilde{f_p}$') #r'$\log_{10} \tilde{f_p}$' + '\n'+
		assert max(np.nanmax(np.log10(np.divide(ND_N_list_nonzero, K**L))), np.nanmax(np.log10(ND_rho_list_nonzero))) <= 0
		print(description_parameters_kbT, 'fraction with f < 0.01 fmax', len([N for N in ND_N_list_nonzero if N < 0.01 * max(ND_N_list_nonzero)])/len(ND_N_list_nonzero))
		if len([i for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] == 0]):
			print('not shown nonzero-freq/norobustness datapoints', len([i for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] == 0]), 'out of', len(ND_rho_list))
		if len([i for i, N in enumerate(ND_N_list) if N/K**L >= ND_rho_list[i]]):
			pheno_vs_num_df = pd.read_csv(filepath + 'data/pheno_vs_integer_'+description_parameters_kbT+'.csv')
			num_vs_pheno = {row['number']: row['phenotype'] for i, row in pheno_vs_num_df.iterrows()}
			print('exceptions for genetic corr', len([i for i, N in enumerate(ND_N_list) if N/K**L >= ND_rho_list[i]]), 'out of', len(ND_rho_list), 'of which zero-rho', len([i for i, N in enumerate(ND_N_list) if N/K**L >= ND_rho_list[i] and ND_rho_list[i] == 0]))
			model_vs_exceptions_g_corr[description_parameters + str(kbT)] = [num_vs_pheno[df_NDGPmap['phenotype'].tolist()[i]] for i, N in enumerate(ND_N_list) if N/K**L >= ND_rho_list[i]]
		###
		ax[columnindex, 3].scatter(ND_rho_list, np.array(NDPevolv_list)/N_pheno, s=3, color=color_list[i], alpha=0.5, lw = 0)
		x, y = zip(*[(rho, ev) for N, rho, ev in zip(ND_N_list, ND_rho_list, NDPevolv_list) if not np.isnan(rho) and not np.isnan(ev) and not np.isnan(N)])
		if len([(rho, ev) for N, rho, ev in zip(ND_N_list, ND_rho_list, NDPevolv_list) if (np.isnan(rho) or np.isnan(ev)) and not np.isnan(N)]):
			raise RuntimeError('found a phenotype with no neutral set size value, but robustness and evolvability')
		if description_parameters.startswith('polyomino'):
			pheno_vs_num_df = pd.read_csv(filepath + 'data/pheno_vs_integer_'+description_parameters_kbT+'.csv')
			pheno_vs_num = {row['phenotype']: row['number'] for i, row in pheno_vs_num_df.iterrows()}
			ph_int = pheno_vs_num['1_1_1']
			print('robust/evolv of single-tile phenotype', [(row['rho'], row['p_evolv']/N_pheno) for i, row in df_NDGPmap.iterrows() if row['phenotype'] == ph_int])
		print('robust/evolv corr', description_parameters, stats.pearsonr(x, y))
		ax[columnindex, 3].set_ylabel( r'$\tilde{\epsilon_p}$') #r'$\tilde{\epsilon_p}$  (normalised)'+
		ax[columnindex, 3].set_xlabel( r'$\tilde{\rho_p}$')#+
		ax[columnindex, 3].set_ylim(0, 1.08)
		ax[columnindex, 3].set_xlim(-0.05, 1)
		assert min(ND_rho_list) >= -0.05 and max(ND_rho_list) <= 1
		assert min(np.array(NDPevolv_list)/N_pheno) >= 0 and max(np.array(NDPevolv_list)/N_pheno) <= 1.08
		ax[columnindex, 3].set_yticks([0, 0.5, 1])
		if description_parameters.startswith('HP'):
			print(description_parameters_kbT, 'min evolv', min(np.array(NDPevolv_list)/N_pheno))
		######
		ax[columnindex, 2].scatter(NDGrobustness.flatten(), NDGevolvability.flatten(), s=2, alpha=0.8, color=color_list[i])
		ax[columnindex, 2].set_ylabel(r'$\tilde{\epsilon_g}$') #r'$\tilde{\epsilon_g}$'+ 
		ax[columnindex, 2].set_xlabel( r'$\tilde{\rho_g}$')#+
		xvalues = np.linspace(0.8 * np.min(NDGrobustness.flatten()), 1, 2)
		ax[columnindex, 2].plot(xvalues, (1-xvalues)* (K-1)*L, color=color_list[i], ls=':', lw=0.7)
		max_gevolv = max(max_gevolv, np.max(NDGevolvability.flatten()))
		for grho, gev in zip(NDGrobustness.flatten(), NDGevolvability.flatten()):
			assert gev <= (1-grho)* (K-1)*L + 0.0001
		#####
		ax[columnindex, 4].legend(handles=custom_lines )
		print('\n')
	ax[columnindex, 2].set_ylim(0, max_gevolv * 1.4)
for i in range(3):
	ax[i, 4].axis('off')
	xlims = ax[i, 1].get_xlim()
	ylims = ax[i, 1].get_ylim()
	xlims_new = (min(xlims[0], ylims[0]) - 0.2, min(0, max(xlims[1], ylims[1]) + 0.2))
	ax[i, 1].plot(np.linspace(xlims_new[0], xlims_new[1]), np.linspace(xlims_new[0], xlims_new[1]), c='k')
	ax[i, 1].set_ylim(xlims_new[0], xlims_new[1])
	ax[i, 1].set_xlim(xlims_new[0], xlims_new[1])
	assert min(min(xlims), min(ylims)) >= min(xlims_new) 
for row in range(3):
	for column in range(4):
		ax[row, column].annotate(r"\textbf{{{x}}}".format(x='ABCD'[row] + '1234'[column]), xy=(0.06, 1.1), xycoords='axes fraction', fontsize=11, weight='bold')
f.tight_layout()
f.savefig(filepath + 'plots/NDGPmap_analysis_biophysical'+'_'.join([param.HP_filename, param.RNA_filename, polyomino_params])+'.png', bbox_inches='tight', dpi=250)
plt.close('all')

###############################################################################################
print('\n\nplot freq distribution for biophysical maps', flush=True)
###############################################################################################
f2, ax2 = plt.subplots(ncols=3, figsize=(8, 3.8))
for columnindex, description_parameters in enumerate(list_to_plot):
	custom_lines, K, L, color_list, label_list = set_up_analysis(type_map_vs_kbT_list[description_parameters], description_parameters)
	ax2[columnindex].set_title(description_param_to_title[description_parameters])
	for i, kbT in enumerate(type_map_vs_kbT_list[description_parameters]):
		if description_parameters == 'polyomino':
			description_parameters_kbT = description_parameters + str(kbT) + polyomino_params
			K, L = int(kbT.split('_')[1]), int(kbT.split('_')[0]) * 4
		else:
			description_parameters_kbT = description_parameters +'_kbT'+str(kbT)
		NDGPmapdata_filename = filepath + 'data/NDGPmapproperties_'+description_parameters_kbT+'.csv'
		try:
			df_NDGPmap = pd.read_csv(NDGPmapdata_filename)
			ND_N_list = df_NDGPmap['neutral set size'].tolist()
		except IOError:
			print('not found', NDGPmapdata_filename)
			continue
		#
		logNdata = np.log10(np.divide([N for N in ND_N_list if N > 0], K**L))
		bins = np.linspace(np.log10(type_map_vs_minmax_freq[description_parameters][0]/K**L), np.log10(type_map_vs_minmax_freq[description_parameters][1]/K**L), 15)
		hist_normalised(logNdata, bins=bins, ax=ax2[columnindex], color=color_list[i])
		ax2[columnindex].legend(handles=custom_lines, loc='lower center', bbox_to_anchor=(0.5, 1.17), ncols=2)
		ax2[columnindex].set_xlabel(r'$\log_{10} \tilde{f_p}$')
		ax2[columnindex].set_ylabel('fraction of\nphenotypes') 
f2.tight_layout(rect=[0,0,1, 0.8])
f2.savefig(filepath + 'plots/freq_distr'+'_'.join([param.HP_filename, param.RNA_filename, polyomino_params])+'.png', bbox_inches='tight', dpi=250)
plt.close('all')




###############################################################################################
print('\n\nplot correlation as alternative way of seeing correlations', flush=True)
###############################################################################################
f4, ax4 = plt.subplots(ncols=3, figsize=(9, 4))
for columnindex, description_parameters in enumerate(list_to_plot):
	custom_lines, K, L, color_list, label_list = set_up_analysis(type_map_vs_kbT_list[description_parameters], description_parameters)
	df_stats_all_versions = {'Pearson correlation': [], 'map parameter': [], 'pair': []}
	ax4[columnindex].set_title(description_param_to_title[description_parameters])
	###
	for i, kbT in enumerate(type_map_vs_kbT_list[description_parameters]):
		if description_parameters == 'polyomino':
			description_parameters_kbT = description_parameters + str(kbT) + polyomino_params
			K, L = int(kbT.split('_')[1]), int(kbT.split('_')[0]) * 4
		else:
			description_parameters_kbT = description_parameters +'_kbT'+str(kbT)
		try:
			df_stats = pd.read_csv(filepath + 'data/NDGPmap_genetic_corr_stats'+description_parameters_kbT+'_perpheno.csv')
		except IOError:
			print('not found', filepath + 'data/NDGPmap_genetic_corr_stats'+description_parameters_kbT+'_perpheno.csv')
			continue
		ax4[columnindex].scatter(df_stats['Pearson correlation neighbours'].tolist(), df_stats['Pearson correlation random'].tolist(), color=color_list[i], s=3, alpha=0.7)
		ax4[columnindex].set_xlabel("Pearson correlation neighbours $g$ and $g'$")
		ax4[columnindex].set_ylabel('Pearson correlation null model $g$ and $h$')
		###
		NDGPmapdata_filename = filepath + 'data/NDGPmapproperties_'+description_parameters_kbT+'.csv'
		try:
			df_NDGPmap = pd.read_csv(NDGPmapdata_filename)
		except IOError:
			print('not found', NDGPmapdata_filename)
		phenotypes_lower_null_model = [p for p, s, t in zip(df_stats['phenotype'].tolist(), df_stats['Pearson correlation neighbours'].tolist(), df_stats['Pearson correlation random'].tolist()) if s <= t + 0.0001 or s < 0 or np.isnan(s) or np.isnan(t)]
		if len(phenotypes_lower_null_model ) > 0:
			print(description_parameters_kbT, 'phenotypes with correlation null model higher than data', 
					len(list(set(phenotypes_lower_null_model))), 'out of which', 
					len([p for p, rho in zip(df_NDGPmap['phenotype'].tolist(), df_NDGPmap['rho'].tolist()) if rho == 0 and p in phenotypes_lower_null_model]), 'are same as zero-robustness phenotypes',
					'and max f is ', max([f/K**L for p, f in zip(df_NDGPmap['phenotype'].tolist(), df_NDGPmap['neutral set size'].tolist()) if p in phenotypes_lower_null_model]))
		pheno_vs_corr, pheno_vs_corr_null = {p: c for p, c in  zip(df_stats['phenotype'].tolist(), df_stats['Pearson correlation neighbours'].tolist())}, {p: c for p, c in  zip(df_stats['phenotype'].tolist(), df_stats['Pearson correlation random'].tolist())}
	ax4[columnindex].legend(handles=custom_lines, loc='lower center', bbox_to_anchor=(0.5, 1.12), ncols=2)  
	ax4[columnindex].set_title(description_param_to_title[description_parameters])
	ax4[columnindex].plot([0, 1], [0, 1], c='k')
f4.tight_layout(rect=[0,0,1, 0.8])
f4.savefig(filepath + 'plots/genetic_corr_alternative'+'_'.join([param.HP_filename, param.RNA_filename, polyomino_params])+'_2.png', bbox_inches='tight', dpi=250)
plt.close('all')
###############################################################################################
print('\n\nplot robustness analysis', flush=True)
###############################################################################################
f3, ax3 = plt.subplots(ncols=3, figsize=(9, 4))
f8, ax8 = plt.subplots(ncols=3, nrows=2, figsize=(9, 6))
for columnindex, description_parameters in enumerate(list_to_plot):
	custom_lines, K, L, color_list, label_list = set_up_analysis(type_map_vs_kbT_list[description_parameters], description_parameters)
	###
	ax3[columnindex].set_title(description_param_to_title[description_parameters])
	###
	for i, kbT in enumerate(type_map_vs_kbT_list[description_parameters]):
		if description_parameters == 'polyomino':
			description_parameters_kbT = description_parameters + str(kbT) + polyomino_params
			K, L = int(kbT.split('_')[1]), int(kbT.split('_')[0]) * 4
		else:
			description_parameters_kbT = description_parameters +'_kbT'+str(kbT)
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
		######
		bins = np.linspace(np.log10(type_map_vs_minmax_rho_freq_ratio[description_parameters][0]), np.log10(type_map_vs_minmax_rho_freq_ratio[description_parameters][1]), 20)
		phenos_with_rho_lower_f = [(N/K**L, ND_rho_list[i]) for i, N in enumerate(ND_N_list) if ND_rho_list[i]/(N/K**L) <= 1 ]
		if len(phenos_with_rho_lower_f):
			print(description_parameters, len(phenos_with_rho_lower_f), 'phenotypes with rho <= f have max rho',  max([x[1] for x in phenos_with_rho_lower_f]), 'max f', max([x[0] for x in phenos_with_rho_lower_f]))
		log_rho_f_ratio = np.log10([ND_rho_list[i]/(N/K**L) for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] > 0])
		assert min(bins) <= min(log_rho_f_ratio) and max(bins) >= max(log_rho_f_ratio)
		assert len([i for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] > 0 and ND_rho_list[i]/(N/K**L) <= 1]) == 0
		max_hist = hist_normalised(log_rho_f_ratio, bins, ax3[columnindex], color_list[i])
		freq_zero_rho_phenos = [N/K**L for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] == 0]
		if len(freq_zero_rho_phenos) > 0:
			print(description_parameters_kbT, len(freq_zero_rho_phenos), 'zero-robustness phenotypes with max freq', max(freq_zero_rho_phenos), 
				'and total freq', np.sum(freq_zero_rho_phenos), 'out of a total of', len([N for N in ND_N_list if not np.isnan(N) and N > 0]), 'phenotypes',
				'with total freq', np.sum([N/K**L for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] > 0]))
		ND_N_list_nonzero = [N for N in ND_N_list if N > 0]
		assert len(freq_zero_rho_phenos) == len(ND_N_list_nonzero) - len(log_rho_f_ratio) and min(log_rho_f_ratio) > 0 and np.nan not in log_rho_f_ratio
		#	ax3[columnindex].set_title('excluded '+ r' phenot. with $\tilde{\rho_p} = 0$', fontsize = 12)
		
		ax3[columnindex].legend(handles=custom_lines, loc='lower center', bbox_to_anchor=(0.5, 1.12), ncols=2)  
		ax3[columnindex].set_xlabel(r'$\log_{10} (\tilde{\rho_p}/\tilde{f_p})$')
		ax3[columnindex].set_ylabel('fraction of\nphenotypes') 
		#ax3[columnindex].plot([0, 0], [0, max_hist/len(log_rho_f_ratio) * 1.5], c='k')
		#if max_hist > 0:
		#	ax3[columnindex].set_ylim(0, max_hist/len(log_rho_f_ratio) * 1.2)
		#####
		ax8[0, columnindex].set_xlabel(r'$\log_{10} \tilde{f_p}$')
		ax8[0, columnindex].set_ylabel(r'$\log_{10} \tilde{\rho_p}$') 
		rho, f, entropy = zip(*[(ND_rho_list[i], N/K**L, ND_entropy_list[i]) for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] > 0])
		ax8[0, columnindex].scatter(np.log10(f), np.log10(rho), color=color_list[i], s=3, alpha=0.7)
		xlims = ax8[0, columnindex].get_xlim()
		logx_values = np.linspace(min(xlims), max(xlims), num=10**4)
		x_values = np.power(10, logx_values)
		upper_bound1 = [x*K**(L-1)/L if x < K**(1-L) else np.nan for x in x_values]
		upper_bound2 = [np.nan if x < K**(1-L) else 1 + np.log(x)/(L * np.log(K)) for x in x_values]
		if description_parameters == 'polyomino':
			color_upper_bound = color_list[i]
		else:
			color_upper_bound = 'grey'
		ax8[0, columnindex].plot(np.log10(x_values), np.log10(upper_bound1), c=color_upper_bound, zorder=-1, lw=0.9)
		ax8[0, columnindex].plot(np.log10(x_values), np.log10(upper_bound2), c=color_upper_bound, zorder=-1, lw=0.9)
		for x, y in zip(f, rho):
			if x < K**(1-L):
				assert y < x*K**(L-1)/L
			else:
				assert y < 1 + np.log(x)/(L * np.log(K))
		###
		predicted_robustness = [K**L*f*ent*np.exp(-1*ent)/(L * np.log(K)) for f, ent in zip(f, entropy)]
		ax8[1, columnindex].set_xlabel(r'$\tilde{\rho_p}$')
		ax8[1, columnindex].set_ylabel(r'predicted $\tilde{\rho_p}$' + '\nfrom entropy and freq.')    
		ax8[1, columnindex].scatter(rho, predicted_robustness, color=color_list[i], s=3, alpha=0.7)  
		print(description_parameters_kbT, 'correlation rho, Sapptington&Mohanty prediction', pearsonr(rho, predicted_robustness))
		ax8[1, columnindex].plot([0, 1], [0, 1], c='k', zorder=-1, lw=0.5)
		ax8[1, columnindex].set_xlim(0, 1)
		assert min(rho) >= 0 and max(rho) <= 1
		ax8[1, columnindex].set_ylim(0, 1)
		assert min(predicted_robustness) >= 0 and max(predicted_robustness) <= 1
		#print(description_parameters_kbT, '\nMohanty prediction pearson corr', pearsonr(rho, predicted_robustness))
		#print('log-log-f-rho pearson corr', pearsonr(np.log(rho), np.log(f)), '\n') 
	ax8[0, columnindex].legend(handles=custom_lines, loc='lower center', bbox_to_anchor=(0.5, 1.12), ncols=2)  
	ax8[0, columnindex].set_title(description_param_to_title[description_parameters])
	###
	ylims = ax3[columnindex].get_ylim()
	ax3[columnindex].plot([0, 0], ylims, c='k')
	ax3[columnindex].set_ylim(ylims)


f3.tight_layout(rect=[0,0,1, 0.8])
f3.savefig(filepath + 'plots/'+'full'+'rho_freq_ratio'+'_'.join([param.HP_filename, param.RNA_filename,  polyomino_params])+'.png', bbox_inches='tight', dpi=250)
f8.tight_layout(rect=[0,0,1, 0.8])
f8.savefig(filepath + 'plots/'+'full'+'Mohanty_analysis'+'_'.join([param.HP_filename, param.RNA_filename,  polyomino_params])+'.png', bbox_inches='tight', dpi=250)

plt.close('all')

###############################################################################################
print('\n\nplot distances between peaks', flush=True)
###############################################################################################
f7, ax7 = plt.subplots(ncols=3, figsize=(9, 4))
for columnindex, description_parameters in enumerate(list_to_plot):
	custom_lines, K, L, color_list, label_list = set_up_analysis(type_map_vs_kbT_list[description_parameters], description_parameters)
	ax7[columnindex].set_title(description_param_to_title[description_parameters])
	for i, kbT in enumerate(type_map_vs_kbT_list[description_parameters]):
		if description_parameters == 'polyomino':
			description_parameters_kbT = description_parameters + str(kbT) + polyomino_params
			K, L = int(kbT.split('_')[1]), int(kbT.split('_')[0]) * 4
		else:
			description_parameters_kbT = description_parameters +'_kbT'+str(kbT)
		######
		try:
			list_distances_peaks = np.load(filepath + 'data/list_distances_peaks'+description_parameters_kbT+'.npy')
		except FileNotFoundError:
			print('not found', filepath + 'data/list_distances_peaks'+description_parameters_kbT+'.npy')
			continue
		Hamming_dist_range = [H - 0.5 for H in range(L + 2)]
		assert max(Hamming_dist_range) >= max(list_distances_peaks) and min(Hamming_dist_range) <= min(list_distances_peaks)
		ax7[columnindex].hist(list_distances_peaks, bins=Hamming_dist_range, density=False, color=color_list[i], alpha=0.3, edgecolor=color_list[i])
		ax7[columnindex].plot(list(range(L + 1)), [len(list_distances_peaks)*comb(L, H)*(K-1)**H/float(K**L) for H in range(L + 1)], c='k')
		ax7[columnindex].set_xlabel('distance between genotypes\nmaximising 2 phenotypes\np and q')
		ax7[columnindex].set_ylabel('frequency')
		ax7[columnindex].legend(handles=custom_lines, loc='lower center', bbox_to_anchor=(0.5, 1.25), ncols=2)  
		###

f7.tight_layout(rect=[0,0,1, 0.8])
f7.savefig(filepath + 'plots/list_distances_peaks'+'_'.join([param.HP_filename, param.RNA_filename,  polyomino_params])+'.png', bbox_inches='tight', dpi=250)

plt.close('all')
###############################################################################################
print('\n\nprint stats for HP', flush=True)
###############################################################################################
GPmap = np.load(filepath + 'data/GPmap_'+'HP16_2Li'+'.npy')
print('fraction of undefined genotypes in GP map', (GPmap < 0.5).sum()/2**16)
###############################################################################################
print('\n\nplot data from Jouffrey analysis', flush=True)
###############################################################################################
f, ax = plt.subplots(nrows=2, ncols=4, figsize=(11, 5.3))
type_map_vs_kbT_list['synthetic_normal_100_15_2'] = (0.05, 0.1, 0.5, 1, 5)
for columnindex, description_parameters in enumerate(['polyomino', 'synthetic_normal_100_15_2']):
	custom_lines, K, L, color_list, label_list = set_up_analysis(type_map_vs_kbT_list[description_parameters], description_parameters)
	for i, kbT in enumerate(type_map_vs_kbT_list[description_parameters]):
		if description_parameters == 'polyomino':
			description_parameters_kbT = description_parameters + str(kbT) + polyomino_params
			K, L = int(kbT.split('_')[1]), int(kbT.split('_')[0]) * 4
		elif description_parameters == 'synthetic_normal_100_15_2':
			description_parameters_kbT = description_parameters +'_kbT'+str(kbT)
			K, L = 2, 15
		cutoff1, cutoff2 = 0.1, 0.25
		try:
			df_jouffrey1 = pd.read_csv(filepath + 'data/jouffreyGPmapdata_filename'+description_parameters_kbT + 'threshold_phenotype_set' + str(cutoff1)+'.csv' )
			df_jouffrey2 = pd.read_csv(filepath + 'data/jouffreyGPmapdata_filename'+description_parameters_kbT + 'threshold_phenotype_set' + str(cutoff2)+'.csv' )
			###
			ax[columnindex, 0].scatter(df_jouffrey1['setrob'].tolist(), df_jouffrey2['setrob'].tolist(), color=color_list[i], s=4, alpha=0.5, lw = 0)
			ax[columnindex, 0].set_xlabel('set-robustness\nthreshold t='+str(cutoff1))
			ax[columnindex, 0].set_ylabel(r"\Large \textbf{{{x}}}".format(x=['A)', 'B)'][columnindex] + description_param_to_title[description_parameters])+'\n\nset-robustness\nthreshold='+str(cutoff2))
			###
			ax[columnindex, 1].scatter(df_jouffrey1['setevolv'].tolist(), df_jouffrey2['setevolv'].tolist(), color=color_list[i], s=4, alpha=0.5, lw = 0)
			ax[columnindex, 1].set_xlabel('set-evolvability\nthreshold t='+str(cutoff1))
			ax[columnindex, 1].set_ylabel('set-evolvability\nthreshold t='+str(cutoff2))

			###
			ax[columnindex, 2].scatter(df_jouffrey1['setrobustevolv'].tolist(), df_jouffrey2['setrobustevolv'].tolist(), color=color_list[i], s=4, alpha=0.5, lw = 0)
			ax[columnindex, 2].set_xlabel('set-robus.-evolv.\nthreshold t='+str(cutoff1))
			ax[columnindex, 2].set_ylabel('set-robus.-evolv.\nthreshold t='+str(cutoff2))
			#####
			if len([(x, y) for x, y in zip(df_jouffrey1['setrobustevolv'].tolist(), df_jouffrey2['setrobustevolv'].tolist()) if not np.isnan(x) and not np.isnan(y) and x > 0 and y > 0]):
				x, y = zip(*[(x, y) for x, y in zip(df_jouffrey1['setrobustevolv'].tolist(), df_jouffrey2['setrobustevolv'].tolist()) if not np.isnan(x) and not np.isnan(y)])
				print(description_parameters_kbT, 'setrobustevolv correlation', pearsonr(x, y))
			else:
				print('no data',description_parameters_kbT )
		except IOError:
			print(filepath + 'data/jouffreyGPmapdata_filename'+description_parameters_kbT + 'threshold_phenotype_set' + str(cutoff1)+'.csv')
	ax[columnindex, -1].legend(handles=custom_lines)
for i in range(2):
	ax[i, -1].axis('off')
	for j in range(3):
		ax[i, j].plot([0, 1], [0, 1], c='k', lw=0.5, zorder=-3)
for row in range(2):
   for column in range(3):
      ax[row, column].annotate(r"\textbf{{{x}}}".format(x='ABCDE'[row] + '12345'[column]), xy=(0.06, 1.1), xycoords='axes fraction', fontsize=11, weight='bold')

f.tight_layout()
f.savefig(filepath + 'plots/'+'Jouffrey_analysisthreshold_dependence'+'_'.join([param.HP_filename, param.RNA_filename,  polyomino_params])+'.png', bbox_inches='tight', dpi=250)
plt.close('all')

###############################################################################################
print('\n\nPolyomino GP map with multiple cutoffs', flush=True)
###############################################################################################
f, ax = plt.subplots(ncols = 5, nrows=3, figsize=(11, 6), width_ratios=[1, 1, 1, 1, 0.5])
polyomino_threshold_list = [3, 5, 10]
description_parameters = 'polyomino'
custom_lines, K, L, color_list, label_list = set_up_analysis(type_map_vs_kbT_list[description_parameters], description_parameters)
for columnindex, polyomino_threshold in enumerate(polyomino_threshold_list):
	max_gevolv = 0.1
	for i, kbT in enumerate(type_map_vs_kbT_list[description_parameters]):
		description_parameters_kbT = description_parameters + str(kbT) + '_nruns'+str(param.polyomino_n_runs)+'_threshold'+str(polyomino_threshold) + '_'+ '_'.join(param.polyomino_filename.split('_')[4:])
		K, L = int(kbT.split('_')[1]), int(kbT.split('_')[0]) * 4
		NDGPmapdata_filename, NDGrobustness_filename, NDGevolvability_filename = filepath + 'data/NDGPmapproperties_'+description_parameters_kbT+'.csv', filepath + 'data/NDGrobustness_'+description_parameters_kbT+'.npy', filepath + 'data/NDGevolvability_'+description_parameters_kbT+'.npy'
		try:
			df_NDGPmap = pd.read_csv(NDGPmapdata_filename)
		except IOError:
			print('not found', NDGPmapdata_filename)
			continue
		ND_N_list = df_NDGPmap['neutral set size'].tolist()
		ND_rho_list = df_NDGPmap['rho'].tolist()
		NDPevolv_list = df_NDGPmap['p_evolv'].tolist()
		NDGrobustness =  np.load(NDGrobustness_filename)
		NDGevolvability = np.load(NDGevolvability_filename)
		##
		ND_N_list_nonzero = [N for N in ND_N_list if not np.isnan(N) and N > 0]
		N_pheno = len(ND_N_list_nonzero)
		ax[columnindex, 0].scatter(np.log10(np.arange(1, len(ND_N_list_nonzero) + 1)), np.log10(sorted(np.divide(ND_N_list_nonzero, K**L), reverse=True)), color=color_list[i], s=4, alpha=0.5, lw = 0)
		ax[columnindex, 0].set_ylabel(r'\Large \textbf{{{x}}}'.format(x='ABCD'[columnindex]+') '+'cut-off x = ' + str(polyomino_threshold))+'\n\n'+r'$\log_{10} \tilde{f_p}$')
		ax[columnindex, 0].set_xlabel(r'$\log_{10}$ freq. rank')
		ax[columnindex, 0].set_xticks([0, 1, 2])
		###
		ND_N_list_nonzero, ND_rho_list_nonzero = [N for i, N in enumerate(ND_N_list) if not np.isnan(N) and N > 0 and ND_rho_list[i] > 0], [N for i, N in enumerate(ND_rho_list) if not np.isnan(ND_N_list[i]) and N > 0 and ND_N_list[i] > 0]
		ax[columnindex, 1].scatter(np.log10(np.divide(ND_N_list_nonzero, K**L)), np.log10(ND_rho_list_nonzero), color=color_list[i], s=4, alpha=0.5, lw = 0)
		ax[columnindex, 1].set_ylabel(r'$\log_{10} \tilde{\rho_p}$')
		xlims = ax[columnindex, 1].get_xlim()
		ylims = ax[columnindex, 1].get_ylim()
		xlims = (min(xlims[0], ylims[0]), max(xlims[1], ylims[1]))
		ax[columnindex, 1].plot(np.linspace(xlims[0], xlims[1]), np.linspace(xlims[0], xlims[1]), c='k')
		ax[columnindex, 1].set_ylim(xlims[0], xlims[1])
		ax[columnindex, 1].set_xlim(xlims[0], xlims[1])
		assert xlims[0] <= min(np.log10(np.divide(ND_N_list_nonzero, K**L))) and xlims[1] >= max(np.log10(np.divide(ND_N_list_nonzero, K**L)))
		assert xlims[0] <= min(np.log10(ND_rho_list_nonzero)) and xlims[1] >= max(np.log10(ND_rho_list_nonzero))
		ax[columnindex, 1].set_xlabel(r'$\log_{10} \tilde{f_p}$' )
		if len([i for i, N in enumerate(ND_N_list) if N/K**L >= ND_rho_list[i]]):
			print(description_parameters_kbT, 'exceptions for genetic corr', len([i for i, N in enumerate(ND_N_list) if N/K**L >= ND_rho_list[i]]), 'out of', len(ND_rho_list), 'of which zero-rho', len([i for i, N in enumerate(ND_N_list) if N/K**L >= ND_rho_list[i] and ND_rho_list[i] == 0]))
		if description_parameters + str(kbT) in model_vs_exceptions_g_corr and polyomino_threshold < param.polyomino_threshold:
			list_phenos_without_corr_standard_cutoff = model_vs_exceptions_g_corr[description_parameters + str(kbT)]		
			pheno_vs_num_df = pd.read_csv(filepath + 'data/pheno_vs_integer_'+description_parameters_kbT+'.csv')
			num_vs_pheno = {row['number']: row['phenotype'] for i, row in pheno_vs_num_df.iterrows()}
			list_phenos_without_corr_lower_cutoff = [num_vs_pheno[df_NDGPmap['phenotype'].tolist()[i]] for i, N in enumerate(ND_N_list) if N/K**L >= ND_rho_list[i]]
			print('no phenotypes for which zero-rho resolved', len([p for p in list_phenos_without_corr_standard_cutoff if p in num_vs_pheno.values() and p not in list_phenos_without_corr_lower_cutoff]), [p for p in list_phenos_without_corr_standard_cutoff if p not in list_phenos_without_corr_lower_cutoff])
		###
		ax[columnindex, 3].scatter(ND_rho_list, np.array(NDPevolv_list)/N_pheno, s=3, color=color_list[i], alpha=0.5, lw = 0)
		ax[columnindex, 3].set_ylabel(r'$\tilde{\epsilon_p}$  (normalised)')
		ax[columnindex, 3].set_xlabel(r'$\tilde{\rho_p}$')
		ax[columnindex, 3].set_ylim(0, 1.08)
		ax[columnindex, 3].set_xlim(-0.05, 1)
		assert -0.05 <= min(ND_rho_list) and 1 >= max(ND_rho_list)
		assert 0 <= min(np.array(NDPevolv_list)/N_pheno) and 1.08 >= max(np.array(NDPevolv_list)/N_pheno)
		ax[columnindex, 3].set_yticks([0, 0.5, 1])
		######
		ax[columnindex, 2].scatter(NDGrobustness.flatten(), NDGevolvability.flatten(), s=3, alpha=0.8, color=color_list[i])
		ax[columnindex, 2].set_ylabel(r'$\tilde{\epsilon_g}$')
		ax[columnindex, 2].set_xlabel(r'$\tilde{\rho_g}$')
		xvalues = np.linspace(0.8 * np.min(NDGrobustness.flatten()), 1, 2)
		ax[columnindex, 2].plot(xvalues, (1-xvalues)* (K-1)*L, color=color_list[i], ls=':', lw=0.7)
		max_gevolv = max(max_gevolv, np.max(NDGevolvability.flatten()))
		for grho, gev in zip(NDGrobustness.flatten(), NDGevolvability.flatten()):
			assert gev <= (1-grho)* (K-1)*L + 0.0001
		#####
	ax[columnindex, 4].legend(handles=custom_lines)
	ax[columnindex, 2].set_ylim(0, max_gevolv * 1.4)
for i in range(3):
	ax[i, 4].axis('off')
for row in range(3):
	for column in range(4):
		ax[row, column].annotate(r"\textbf{{{x}}}".format(x='ABCD'[row] + '1234'[column]), xy=(0.06, 1.1), xycoords='axes fraction', fontsize=11, weight='bold')
f.tight_layout()
f.savefig(filepath + 'plots/NDGPmap_analysis_biophysicalpolyomino'+'_nruns'+str(param.polyomino_n_runs) + '_'+ '_'.join(param.polyomino_filename.split('_')[4:])+'.png', bbox_inches='tight', dpi=250)
plt.close('all')

