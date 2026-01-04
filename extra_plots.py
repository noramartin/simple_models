import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import isfile
from scipy import stats as stats
import pandas as pd
from matplotlib.lines import Line2D
import parameters as param
import json

plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[cm]{sfmath}'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'cm'


##############################################################################################
polyomino_kbT_list = ['2_8', '3_3']
polyomino_params = '_'+ '_'.join(param.polyomino_filename.split('_')[2:])
polyomino_colors = matplotlib.cm.tab10(range(10))[:len(polyomino_kbT_list)]
polyomino_label_list = ['$S_{%s, %s}$' %(kbT.split('_')[0], kbT.split('_')[1]) for i, kbT in enumerate(polyomino_kbT_list)]
polyomino_legend = [Line2D([0], [0], mfc=polyomino_colors[i], ls='', marker='o', label=polyomino_label_list[i], mew=0, ms=5) for i in range(len(polyomino_colors))]

###############################################################################################
print('\n\nPolyomino GP map with multiple cutoffs', flush=True)
###############################################################################################
polyomino_threshold_list = [25, 50, 100]
f, ax = plt.subplots(ncols = 5, nrows=3, figsize=(10, 6.5), width_ratios=[1, 1, 1, 1, 0.5])
for columnindex, polyomino_threshold in enumerate(polyomino_threshold_list):
	max_gevolv = 0.1
	for i, kbT in enumerate(polyomino_kbT_list):
		description_parameters_kbT = 'polyomino' + str(kbT) + '_nruns'+str(param.polyomino_n_runs)+'_threshold'+str(polyomino_threshold) + '_'+ '_'.join(param.polyomino_filename.split('_')[4:])
		K, L = int(kbT.split('_')[1]), int(kbT.split('_')[0]) * 4
		NDGPmapdata_filename, NDGrobustness_filename, NDGevolvability_filename = './data/NDGPmapproperties_'+description_parameters_kbT+'.csv', './data/NDGrobustness_'+description_parameters_kbT+'.npy', './data/NDGevolvability_'+description_parameters_kbT+'.npy'
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
		ax[columnindex, 0].scatter(np.log10(np.arange(1, len(ND_N_list_nonzero) + 1)), np.log10(sorted(np.divide(ND_N_list_nonzero, K**L), reverse=True)), color=polyomino_colors[i], s=4, lw = 0)
		ax[columnindex, 0].set_ylabel(r'$\log_{10} \tilde{f_p}$', fontsize=13)
		ax[columnindex, 0].set_xlabel(r'$\log_{10}$ freq. rank', fontsize=13)
		ax[columnindex, 0].set_xticks([0, 1, 2])
		###
		ND_N_list_nonzero, ND_rho_list_nonzero = [N for i, N in enumerate(ND_N_list) if not np.isnan(N) and N > 0 and ND_rho_list[i] > 0], [N for i, N in enumerate(ND_rho_list) if not np.isnan(ND_N_list[i]) and N > 0 and ND_N_list[i] > 0]
		ax[columnindex, 1].scatter(np.log10(np.divide(ND_N_list_nonzero, K**L)), np.log10(ND_rho_list_nonzero), color=polyomino_colors[i], s=4, lw = 0)
		ax[columnindex, 1].set_ylabel(r'$\log_{10} \tilde{\rho_p}$', fontsize=13)

		#assert xlims[0] <= min(np.log10(np.divide(ND_N_list_nonzero, K**L))) and xlims[1] >= max(np.log10(np.divide(ND_N_list_nonzero, K**L)))
		#assert xlims[0] <= min(np.log10(ND_rho_list_nonzero)) and xlims[1] >= max(np.log10(ND_rho_list_nonzero))
		ax[columnindex, 1].set_xlabel(r'$\log_{10} \tilde{f_p}$' , fontsize=13)
		if polyomino_threshold < param.polyomino_threshold:
			try:
				with open('./data/exceptions_g_corr'+description_parameters_kbT+'.csv', 'r') as fp:
					list_phenos_without_corr_standard_cutoff = json.load(fp)
				pheno_vs_num_df = pd.read_csv('./data/pheno_vs_integer_'+description_parameters_kbT+'.csv')
				num_vs_pheno = {row['number']: row['phenotype'] for i, row in pheno_vs_num_df.iterrows()}
				list_phenos_with_corr_lower_cutoff = [num_vs_pheno[df_NDGPmap['phenotype'].tolist()[i]] for i, N in enumerate(ND_N_list) if N/K**L < ND_rho_list[i] and N > 0]
				list_phenos_corr_resolved = [p for p in list_phenos_without_corr_standard_cutoff if p in list_phenos_with_corr_lower_cutoff]
				print('no phenotypes for which zero-rho resolved', len(list_phenos_corr_resolved), list_phenos_corr_resolved)
			except IOError:
				pass
		###
		ax[columnindex, 3].scatter(ND_rho_list, np.array(NDPevolv_list)/N_pheno, s=3, color=polyomino_colors[i], lw = 0)
		ax[columnindex, 3].set_ylabel(r'$\tilde{\epsilon_p}$  (normalised)', fontsize=13)
		ax[columnindex, 3].set_xlabel(r'$\tilde{\rho_p}$', fontsize=13)
		ax[columnindex, 3].set_ylim(0, 1.08)
		ax[columnindex, 3].set_xlim(-0.05, 1)
		assert -0.05 <= min(ND_rho_list) and 1 >= max(ND_rho_list)
		assert 0 <= min(np.array(NDPevolv_list)/N_pheno) and 1.08 >= max(np.array(NDPevolv_list)/N_pheno)
		ax[columnindex, 3].set_yticks([0, 0.5, 1])
		######
		ax[columnindex, 2].scatter(NDGrobustness.flatten(), NDGevolvability.flatten(), s=3, alpha=0.05, color=polyomino_colors[i])
		ax[columnindex, 2].set_ylabel(r'$\tilde{\epsilon_g}$', fontsize=13)
		ax[columnindex, 2].set_xlabel(r'$\tilde{\rho_g}$', fontsize=13)
		xvalues = np.linspace(0.8 * np.min(NDGrobustness.flatten()), 1, 2)
		ax[columnindex, 2].plot(xvalues, (1-xvalues)* (K-1)*L, color=polyomino_colors[i], ls=':', lw=0.7)
		max_gevolv = max(max_gevolv, np.max(NDGevolvability.flatten()))
		for grho, gev in zip(NDGrobustness.flatten(), NDGevolvability.flatten()):
			assert gev <= (1-grho)* (K-1)*L + 0.0001
		#####
	ax[columnindex, 4].legend(handles=polyomino_legend)
	ax[columnindex, 2].set_ylim(0, max_gevolv * 1.4)
	xlims = ax[columnindex, 1].get_xlim()
	ylims = ax[columnindex, 1].get_ylim()
	xlims = (min(xlims[0], ylims[0]), max(xlims[1], ylims[1]))
	ax[columnindex, 1].plot(np.linspace(xlims[0], xlims[1]), np.linspace(xlims[0], xlims[1]), c='k')
	ax[columnindex, 1].set_ylim(xlims[0], xlims[1])
	ax[columnindex, 1].set_xlim(xlims[0], xlims[1])
for i in range(3):
	ax[i, 4].axis('off')
for row in range(3):
	for column in range(4):
		ax[row, column].set_title('ABCD'[row] + '1234'[column] + ') cut-off x = ' + str(polyomino_threshold_list[row]), loc='left')
f.tight_layout()
f.savefig('./plots/NDGPmap_analysis_biophysicalpolyomino_nruns'+str(param.polyomino_n_runs) + '_'+ '_'.join(param.polyomino_filename.split('_')[4:])+'.png', bbox_inches='tight', dpi=250)
plt.close('all')

###############################################################################################
print('\n\nplot data from Jouffrey analysis', flush=True)
###############################################################################################
f, ax = plt.subplots(nrows=2, ncols=4, figsize=(8.5, 4.3), width_ratios=[1, 1, 1, 0.6])
for columnindex, description_parameters in enumerate(['polyomino', 'synthetic_normal_100_15_2']):
	if description_parameters.startswith('synthetic'):
		Tlist = (0.05, 0.1, 0.5, 1)
		color_list = [plt.get_cmap('viridis')(i) for i in np.linspace(0, 1, len(Tlist))]
		legend_elements = [Line2D([0], [0], mfc=color_list[i], ls='', marker='o', label='$T$='+str(kbT), mew=0, ms=5) for i, kbT in enumerate(Tlist)]
	else:
		Tlist = ['2_8', '3_3']
		color_list = polyomino_colors
		legend_elements = [Line2D([0], [0], mfc=polyomino_colors[i], ls='', marker='o', label=polyomino_label_list[i], mew=0, ms=5) for i, kbT in enumerate(Tlist)]
	for i, kbT in enumerate(Tlist):
		if description_parameters == 'polyomino':
			description_parameters_kbT = description_parameters + str(kbT) + polyomino_params
			K, L = int(kbT.split('_')[1]), int(kbT.split('_')[0]) * 4
		elif description_parameters == 'synthetic_normal_100_15_2':
			description_parameters_kbT = description_parameters +'_kbT'+str(kbT)
			K, L = int(description_parameters.split('_')[-1]), int(description_parameters.split('_')[-2])
		cutoff1, cutoff2 = 0.1, 0.25
		try:
			df_jouffrey1 = pd.read_csv('./data/jouffreyGPmapdata_filename'+description_parameters_kbT + 'threshold_phenotype_set' + str(cutoff1)+'.csv' )
			df_jouffrey2 = pd.read_csv('./data/jouffreyGPmapdata_filename'+description_parameters_kbT + 'threshold_phenotype_set' + str(cutoff2)+'.csv' )
			###
			ax[columnindex, 0].scatter(df_jouffrey1['setrob'].tolist(), df_jouffrey2['setrob'].tolist(), color=color_list[i], s=4, lw = 0)
			ax[columnindex, 0].set_xlabel('set-robustness\nthreshold t='+str(cutoff1), fontsize=12)
			ax[columnindex, 0].set_ylabel('set-robustness\nthreshold t='+str(cutoff2), fontsize=12)
			###
			ax[columnindex, 1].scatter(df_jouffrey1['setevolv'].tolist(), df_jouffrey2['setevolv'].tolist(), color=color_list[i], s=4, lw = 0)
			ax[columnindex, 1].set_xlabel('set-evolvability\nthreshold t='+str(cutoff1), fontsize=12)
			ax[columnindex, 1].set_ylabel('set-evolvability\nthreshold t='+str(cutoff2), fontsize=12)

			###
			ax[columnindex, 2].scatter(df_jouffrey1['setrobustevolv'].tolist(), df_jouffrey2['setrobustevolv'].tolist(), color=color_list[i], s=4, lw = 0)
			ax[columnindex, 2].set_xlabel('set-robus.-evolv.\nthreshold t='+str(cutoff1), fontsize=12)
			ax[columnindex, 2].set_ylabel('set-robus.-evolv.\nthreshold t='+str(cutoff2), fontsize=12)
			#####
			if len([(x, y) for x, y in zip(df_jouffrey1['setrobustevolv'].tolist(), df_jouffrey2['setrobustevolv'].tolist()) if not np.isnan(x) and not np.isnan(y) and x > 0 and y > 0]):
				x, y = zip(*[(x, y) for x, y in zip(df_jouffrey1['setrobustevolv'].tolist(), df_jouffrey2['setrobustevolv'].tolist()) if not np.isnan(x) and not np.isnan(y)])
				print(description_parameters_kbT, 'setrobustevolv correlation', stats.pearsonr(x, y))
		except IOError:
			print('./data/jouffreyGPmapdata_filename'+description_parameters_kbT + 'threshold_phenotype_set' + str(cutoff1)+'.csv')
	ax[columnindex, -1].legend(handles=legend_elements)
for i in range(2):
	ax[i, -1].axis('off')
	for j in range(3):
		ax[i, j].plot([0, 1], [0, 1], c='k', lw=0.5, zorder=-3)
for row in range(2):
   for column in range(3):
      ax[row, column].set_title('('+'ABCDE'[row] + '12345'[column] + ') ' + ['Polyomino', 'Synthetic model'][row], loc='left')

f.tight_layout()
f.savefig('./plots/Jouffrey_analysisthreshold_dependence'+polyomino_params+'.png', bbox_inches='tight', dpi=250)
plt.close('all')

###############################################################################################
print('\n\ncheck D map against low-T ND map', flush=True)
###############################################################################################
K_Dmap, L_Dmap = 4, 8
n_list = (50, 100, 500)
color_list = matplotlib.cm.tab10(range(10))[2:2+len(n_list)]
legend_elements = [Line2D([0], [0], mfc=color_list[i], ls='', marker='o', label='$n_p$='+str(N_pheno), mew=0, ms=5) for i, N_pheno in enumerate(n_list)]
f, ax = plt.subplots(ncols=6, nrows=3, figsize=(12, 7), width_ratios=[1,]*5 +[0.5])
N_vs_fit_log = {}
for rowindex, kbT in enumerate([0.0001, 0.1, 1]):
	for i, N_pheno in enumerate(n_list):
	   description_parameters = 'synthetic_normal_' + str(N_pheno)+'_'+ str(L_Dmap) + '_' + str(K_Dmap) 
	   description_parameters_kbT = description_parameters +'_kbT'+str(kbT)
	   DGPmapdata_filename, DGrobustness_filename, DGevolvability_filename = './data/GPmapproperties_'+description_parameters+'.csv', './data/Grobustness_'+description_parameters+'.npy', './data/Gevolvability_'+description_parameters+'.npy'
	   NDGPmapdata_filename, NDGrobustness_filename, NDGevolvability_filename = './data/NDGPmapproperties_'+description_parameters_kbT+'.csv', './data/NDGrobustness_'+description_parameters_kbT+'.npy', './data/NDGevolvability_'+description_parameters_kbT+'.npy'
	   try:
	      df_DGPmap = pd.read_csv(DGPmapdata_filename)
	      df_NDGPmap = pd.read_csv(NDGPmapdata_filename)
	   except IOError:
	      print('not found', DGPmapdata_filename, NDGPmapdata_filename)
	      continue
	   DGrobustness =  np.load(DGrobustness_filename)
	   DGevolvability = np.load(DGevolvability_filename)
	   NDGrobustness =  np.load(NDGrobustness_filename)
	   NDGevolvability = np.load(NDGevolvability_filename)
	   #####
	   for p1, p2 in zip(df_DGPmap['phenotype'].tolist(), df_NDGPmap['phenotype'].tolist()):
	   	assert p1 == p2
	   ######
	   ax[rowindex, 0].scatter(np.divide(df_DGPmap['neutral set size'].tolist(), K_Dmap**L_Dmap), np.divide(df_NDGPmap['neutral set size'].tolist(), K_Dmap**L_Dmap), color=color_list[i], s=4, lw = 0)
	   ax[rowindex, 0].set_ylabel(r'$\tilde{f}_p$', fontsize=14)
	   ax[rowindex, 0].set_xlabel(r'$f_p$', fontsize=13)
	   ax[rowindex, 0].set_yscale('log')
	   ax[rowindex, 0].set_xscale('log')
	   ######
	   ax[rowindex, 1].scatter(df_DGPmap['rho'].tolist(), df_NDGPmap['rho'].tolist(), color=color_list[i], s=4, lw = 0)
	   ax[rowindex, 1].set_ylabel(r'$\tilde{\rho}_p$', fontsize=14)
	   ax[rowindex, 1].set_xlabel(r'$\rho_p$', fontsize=13)
	   ######
	   ax[rowindex, 2].scatter(df_DGPmap['p_evolv'].tolist(), df_NDGPmap['p_evolv'].tolist(), color=color_list[i], s=4, lw = 0)
	   ax[rowindex, 2].set_ylabel(r'$\tilde{e}_p$', fontsize=14)
	   ax[rowindex, 2].set_xlabel(r'$e_p$', fontsize=13)
	   ######
	   ax[rowindex, 3].scatter(DGrobustness.flatten(), NDGrobustness.flatten(), color=color_list[i], s=4, lw = 0)
	   ax[rowindex, 3].set_ylabel(r'$\tilde{\rho}_g$', fontsize=14)
	   ax[rowindex, 3].set_xlabel(r'$\rho_g$', fontsize=13)
	   ######
	   ax[rowindex, 4].scatter(DGevolvability.flatten(), NDGevolvability.flatten(), color=color_list[i], s=4, lw = 0)
	   ax[rowindex, 4].set_ylabel(r'$\tilde{e}_g$', fontsize=14)
	   ax[rowindex, 4].set_xlabel(r'$e_g$', fontsize=13)
	   ###
	ax[rowindex, -1].legend(handles=legend_elements)
	for i in range(5):
		xlims, ylims = ax[rowindex, i].get_xlim(), ax[rowindex, i].get_ylim()
		lims = (min(xlims[0], ylims[0]), max(xlims[1], ylims[1]))
		ax[rowindex, i].set_xlim(lims)
		ax[rowindex, i].set_ylim(lims)
		ax[rowindex, i].plot(lims, lims, c='k', zorder=-4)
		ax[rowindex, i].set_title('kbT='+str(kbT))
	ax[rowindex, -1].axis('off')
f.tight_layout()
f.savefig('./plots/DGPmap_vs_lowTmap'+'.png', bbox_inches='tight', dpi=250)
plt.close('all')



