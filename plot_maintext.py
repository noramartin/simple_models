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
from functions.general_functions import hist_normalised

plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[cm]{sfmath}'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'cm'

type_plot_vs_n_and_kbT_list = {'vary_N_pheno': [(N_pheno, 'mean_energygap') for N_pheno in (50, 100, 500)], 'vary_T': [(100, kbT) for kbT in (0.2, 1, 2.5)]}
name_synthetic = '_15_2'
type_map_vs_kbT_list = {'RNA12': [30, 37, 45], 'polyomino': ['2_8', '3_3'], 'HP16_2Li': [0.001, 0.25, 0.5, 1]}
polyomino_params = '_'+ '_'.join(param.polyomino_filename.split('_')[2:])
model_vs_KL = {'RNA12': (4, 12), name_synthetic: (int(name_synthetic.split('_')[-1]), int(name_synthetic.split('_')[-2])), 'HP16_2Li': (2, 16)}
description_param_to_title = {'RNA12': 'RNA', 'polyomino': 'Polyomino', name_synthetic: 'synthetic', 'HP16_2Li': r'lattice protein'}
model_vs_colors, model_vs_legend_elements = {}, {}
for model in type_map_vs_kbT_list:
   if model.startswith('polyomino'):
      model_vs_colors[model] = matplotlib.cm.tab10(range(10))[:len(type_map_vs_kbT_list[model])]
   else:
      model_vs_colors[model] = [plt.get_cmap('plasma')(i) for i in np.linspace(0, 0.92, len(type_map_vs_kbT_list[model]))] #continuous parameter
   if model.startswith('polyomino'):
      label_list = ['$S_{%s, %s}$' %(kbT.split('_')[0], kbT.split('_')[1]) for i, kbT in enumerate(type_map_vs_kbT_list[model])]
   elif model.startswith('RNA'):
      label_list = ['$T$='+str(kbT)+ r'$^{o}$C' for i, kbT in enumerate(type_map_vs_kbT_list[model])]
   else:
      label_list = ['$k_bT$='+str(kbT) for i, kbT in enumerate(type_map_vs_kbT_list[model])]
   model_vs_legend_elements[model] = [Line2D([0], [0], mfc=model_vs_colors[model][i], ls='', marker='o', label=label_list[i], mew=0, ms=5) for i in range(len(model_vs_colors[model]))]
for type_plot in type_plot_vs_n_and_kbT_list:
   if type_plot == 'vary_N_pheno':
      model_vs_colors[type_plot] = matplotlib.cm.tab10(range(10))[2:2+len(type_plot_vs_n_and_kbT_list[type_plot])]
   else:
      model_vs_colors[type_plot] = [plt.get_cmap('viridis')(i) for i in np.linspace(0, 0.92, len(type_plot_vs_n_and_kbT_list[type_plot]))] #continuous parameter
   if type_plot == 'vary_N_pheno':
      model_vs_legend_elements[type_plot] = [Line2D([0], [0], mfc=model_vs_colors[type_plot][i], ls='', marker='o', label='$n_p$='+str(N_pheno), mew=0, ms=5) for i, (N_pheno, kbT) in enumerate(type_plot_vs_n_and_kbT_list[type_plot])]
   else:
      model_vs_legend_elements[type_plot] = [Line2D([0], [0], mfc=model_vs_colors[type_plot][i], ls='', marker='o', label='$T$='+str(kbT), mew=0, ms=5) for i, (N_pheno, kbT) in enumerate(type_plot_vs_n_and_kbT_list[type_plot])]

################################################################################################
print('\n\nneutral set size plot', flush=True)
###############################################################################################
f, ax = plt.subplots(ncols=5, figsize=(11, 2.7))
for columnindex, (model, type_plot) in enumerate([('RNA12', 'all'), ('polyomino', 'all'), ('HP16_2Li', 'all'), (name_synthetic, 'vary_N_pheno'), (name_synthetic, 'vary_T')]):
   if model == name_synthetic:
      list_Npheno_andT, color_list, legend_elements = type_plot_vs_n_and_kbT_list[type_plot], model_vs_colors[type_plot], model_vs_legend_elements[type_plot]
   else:
      list_Npheno_andT, color_list, legend_elements = [('none', kbT) for kbT in type_map_vs_kbT_list[model]], model_vs_colors[model], model_vs_legend_elements[model]
   for i, (N_pheno, kbT) in enumerate(list_Npheno_andT):
      if model == 'polyomino':
         description_parameters_kbT = model + str(kbT) + polyomino_params
         K, L = int(kbT.split('_')[1]), int(kbT.split('_')[0]) * 4
      elif model == name_synthetic:
         description_parameters_kbT = 'synthetic_normal_' + str(N_pheno) + model + '_kbT' + str(kbT)
      else:
         description_parameters_kbT = model +'_kbT'+str(kbT)
      if model != 'polyomino':
         K, L = model_vs_KL[model]
      NDGPmapdata_filename = './data/NDGPmapproperties_'+description_parameters_kbT+'.csv'
      try:
         df_NDGPmap = pd.read_csv(NDGPmapdata_filename)
      except IOError:
         print('not found', NDGPmapdata_filename)
         continue
      ND_N_list = df_NDGPmap['neutral set size'].tolist()
      ##
      rank_data, N_data_by_rank = np.arange(1, len(ND_N_list) + 1), sorted(ND_N_list, reverse=True)
      ax[columnindex].scatter(np.log10(rank_data), np.log10(np.divide(N_data_by_rank, K**L)), color=color_list[i], s=5, lw = 0)
      ax[columnindex].set_ylabel(r'$\log_{10} \tilde{f}_p$', fontsize=13)
      ax[columnindex].set_xlabel(r'$\log_{10}$ freq. rank', fontsize=13)
      if len([N for N in N_data_by_rank if not np.isnan(N) and N > 0 and N < 0.01*np.nanmax(N_data_by_rank)]) < 0.5* len([N for N in N_data_by_rank if not np.isnan(N) and N > 0]):
         print(description_parameters_kbT, 'ratio highest to lowest freq', np.nanmax(N_data_by_rank)/min([N for N in N_data_by_rank if not np.isnan(N) and N > 0]))
      print(description_parameters_kbT, 'freaction less than 1% of fmax', len([N for N in N_data_by_rank if not np.isnan(N) and N > 0 and N < 0.01*np.nanmax(N_data_by_rank)])/len([N for N in N_data_by_rank if not np.isnan(N) and N > 0]))
      if model == name_synthetic:
         ax[columnindex].set_title('ABCDE'[columnindex] + ') ' + description_param_to_title[model]+' ' + {'vary_N_pheno': r'(vary $n_p$)', 'vary_T': r'(vary $T$)'}[type_plot])
      else:
         ax[columnindex].set_title('ABCDE'[columnindex] + ') '+description_param_to_title[model])
   if type_plot == 'vary_T':
      try:
         df_DGPmap = pd.read_csv('./data/GPmapproperties_synthetic_normal_' + str(N_pheno) + model +'.csv')
         D_f_list = [N/K**L for N in df_DGPmap['neutral set size'].tolist() if not np.isnan(N) and N > 0]
         ax[columnindex].scatter(np.log10(np.arange(1, 1+ len(D_f_list))), np.log10(sorted(D_f_list, reverse=True)), color='grey', s=4, lw = 0, zorder=-3)
         legend_elements = [Line2D([0], [0], mfc='grey', ls='', marker='o', label=r'determin.', mew=0, ms=5),] + legend_elements
      except IOError:
         pass
   if max(ax[columnindex].get_xlim()) < 2.6:
      ax[columnindex].set_xticks([0, 1, 2])
   else:
      ax[columnindex].set_xticks([0, 1, 2, 3])
   abs_ylims = np.abs(ax[columnindex].get_ylim())
   if max(abs_ylims) - min(abs_ylims) < 2:
      ax[columnindex].set_yticks([-1*i for i in np.arange(round(min(abs_ylims), 1) - 0.1, 0.2 + round(max(abs_ylims), 1), step=0.2)])
   elif max(abs_ylims) - min(abs_ylims) < 4:
      ax[columnindex].set_yticks([-1*i for i in range(int(min(abs_ylims)), 1 + int(max(abs_ylims)))])
   else:
      ax[columnindex].set_yticks([-1*i for i in np.arange(int(min(abs_ylims)), 1 + int(max(abs_ylims)), step=2)])
   ax[columnindex].legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
                         ncol=2, fancybox=False, columnspacing=0.5, handletextpad=0.2, borderaxespad=3)
f.tight_layout()
f.savefig('./plots/phenotypic_bias_plot'+'_'.join([param.HP_filename, param.RNA_filename, polyomino_params, name_synthetic])+'.png', bbox_inches='tight', dpi=250)
plt.close('all')
################################################################################################
print('\n\nsmall extra plot - vector norm', flush=True)
###############################################################################################
f, ax = plt.subplots(ncols=2, figsize=(5, 2.9))
for columnindex, (model, type_plot) in enumerate([(name_synthetic, 'vary_N_pheno'), (name_synthetic, 'vary_T')]):
   list_Npheno_andT, color_list, legend_elements = type_plot_vs_n_and_kbT_list[type_plot], model_vs_colors[type_plot], model_vs_legend_elements[type_plot]
   for i, (N_pheno, kbT) in enumerate(list_Npheno_andT):
      description_parameters_kbT = 'synthetic_normal_' + str(N_pheno) + model + '_kbT' + str(kbT)
      K, L = model_vs_KL[model]
      NDGPmapdata_filename = './data/NDGPmapproperties_'+description_parameters_kbT+'.csv'
      try:
         df_NDGPmap = pd.read_csv(NDGPmapdata_filename)
      except IOError:
         print('not found', NDGPmapdata_filename)
         continue
      ND_N_list = df_NDGPmap['neutral set size'].tolist()
      ax[columnindex].scatter(df_NDGPmap['length phenotypic vector'].tolist(), np.log10(np.divide(ND_N_list, K**L)), color=color_list[i], s=5, lw = 0)
      ax[columnindex].set_ylabel(r'$\log_{10} \tilde{f}_p$', fontsize=13)
      ax[columnindex].set_xlabel(r'$\vec{v}_p$ norm', fontsize=13)
      ax[columnindex].set_title('ABCDE'[columnindex] + ') ' + description_param_to_title[model]+' ' + {'vary_N_pheno': r'(vary $n_p$)', 'vary_T': r'(vary $T$)'}[type_plot])
   ax[columnindex].legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
                         ncol=2, fancybox=False, columnspacing=0.5, handletextpad=0.2, borderaxespad=3)
   ax[columnindex].set_xticks([2, 3, 4, 5, 6])
   ax[columnindex].set_yticks([-1, -2, -3, -4])
f.tight_layout()
f.savefig('./plots/vector_norm_plot'+'_'.join([name_synthetic])+'.png', bbox_inches='tight', dpi=300)
plt.close('all')
################################################################################################
print('\n\nsmall extra plot - vector norm 2', flush=True)
###############################################################################################
for cutoff in np.arange(-5, 0):
   f, ax = plt.subplots(ncols=2, figsize=(5, 2.9))
   for columnindex, (model, type_plot) in enumerate([(name_synthetic, 'vary_N_pheno'), (name_synthetic, 'vary_T')]):
      list_Npheno_andT, color_list, legend_elements = type_plot_vs_n_and_kbT_list[type_plot], model_vs_colors[type_plot], model_vs_legend_elements[type_plot]
      for i, (N_pheno, kbT) in enumerate(list_Npheno_andT):
         description_parameters = 'synthetic_normal_' + str(N_pheno) + model 
         K, L = model_vs_KL[model]
         NDGPmapdata_filename = './data/NDGPmapproperties_lowG_'+description_parameters+'.csv'
         try:
            df_NDGPmap = pd.read_csv(NDGPmapdata_filename)
         except IOError:
            print('not found', NDGPmapdata_filename)
            continue
         ND_N_list = df_NDGPmap['# sequences below '+str(cutoff)].tolist()
         ax[columnindex].scatter(df_NDGPmap['length phenotypic vector'].tolist(), np.log10(np.divide(ND_N_list, K**L)), color=color_list[i], s=5, lw = 0)
         ax[columnindex].set_ylabel(r'number seq. $<$ '+str(cutoff), fontsize=13)
         ax[columnindex].set_xlabel(r'$\vec{v}_p$ norm', fontsize=13)
         ax[columnindex].set_title('ABCDE'[columnindex] + ') ' + description_param_to_title[model]+' ' + {'vary_N_pheno': r'(vary $n_p$)', 'vary_T': r'(vary $T$)'}[type_plot])
      ax[columnindex].legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
                            ncol=2, fancybox=False, columnspacing=0.5, handletextpad=0.2, borderaxespad=3)
      ax[columnindex].set_xticks([2, 3, 4, 5, 6])
      ax[columnindex].set_yticks([-1, -2, -3, -4])
   f.tight_layout()
   f.savefig('./plots/vector_norm_plot'+'_'.join([name_synthetic])+'lowG'+str(cutoff)+'.png', bbox_inches='tight', dpi=300)
   plt.close('all')
assert 1 == 2
################################################################################################
print('\n\ngenetic correlations plot - including pearsonr', flush=True)
###############################################################################################
f, ax = plt.subplots(ncols=5, nrows=2, figsize=(11, 5))
for columnindex, (model, type_plot) in enumerate([('RNA12', 'all'), ('polyomino', 'all'), ('HP16_2Li', 'all'), (name_synthetic, 'vary_N_pheno'), (name_synthetic, 'vary_T')]):
   max_cov = 0
   if model == name_synthetic:
      list_Npheno_andT, color_list, legend_elements = type_plot_vs_n_and_kbT_list[type_plot], model_vs_colors[type_plot], model_vs_legend_elements[type_plot]
   else:
      list_Npheno_andT, color_list, legend_elements = [('none', kbT) for kbT in type_map_vs_kbT_list[model]], model_vs_colors[model], model_vs_legend_elements[model]
   for i, (N_pheno, kbT) in enumerate(list_Npheno_andT):
      if model == 'polyomino':
         description_parameters_kbT = model + str(kbT) + polyomino_params
         K, L = int(kbT.split('_')[1]), int(kbT.split('_')[0]) * 4
      elif model == name_synthetic:
         description_parameters_kbT = 'synthetic_normal_' + str(N_pheno) + model + '_kbT' + str(kbT)
      else:
         description_parameters_kbT = model +'_kbT'+str(kbT)
      if model != 'polyomino':
         K, L = model_vs_KL[model]
      try:
         df_NDGPmap = pd.read_csv('./data/NDGPmapproperties_'+description_parameters_kbT+'.csv')
         df_stats = pd.read_csv('./data/NDGPmap_genetic_corr_stats'+description_parameters_kbT+'_perpheno.csv')
      except IOError:
         print('not found', description_parameters_kbT)
         continue
      ND_N_list = df_NDGPmap['neutral set size'].tolist()
      ND_rho_list = df_NDGPmap['rho'].tolist()
      ###
      exceptions_g_corr = [df_NDGPmap['phenotype'].tolist()[i] for i, N in enumerate(ND_N_list) if N/K**L >= ND_rho_list[i]]
      if len(exceptions_g_corr):
         print('\n', description_parameters_kbT, 'exceptions for genetic corr', len(exceptions_g_corr), 'out of', len(ND_rho_list), 'of which zero-rho', len([i for i, N in enumerate(ND_N_list) if N/K**L >= ND_rho_list[i] and ND_rho_list[i] == 0]))
      if model == 'polyomino':
         pheno_vs_num_df = pd.read_csv('./data/pheno_vs_integer_'+description_parameters_kbT+'.csv')
         num_vs_pheno = {row['number']: row['phenotype'] for i, row in pheno_vs_num_df.iterrows()}
         exceptions_g_corr_p = [num_vs_pheno[p] for p in exceptions_g_corr]
         if len(exceptions_g_corr):
            print('total freq of all exceptions compared to all', sum([N/K**L for i, N in enumerate(ND_N_list) if N/K**L >= ND_rho_list[i]])/sum([N/K**L for i, N in enumerate(ND_N_list) if not np.isnan(N)]))
         with open('./data/exceptions_g_corr'+description_parameters_kbT+'.csv', 'w') as fp:
            json.dump(exceptions_g_corr_p, fp)
      ##
      nonzero_N, nonzero_rho = zip(*[(N, rho) for N, rho in zip(ND_N_list, ND_rho_list) if not np.isnan(N) and not np.isnan(rho) and N > 0 and rho > 0])
      ax[0, columnindex].scatter(np.log10(np.divide(nonzero_N, K**L)), np.log10(nonzero_rho), color=color_list[i],  s=5, lw = 0)
      ax[0, columnindex].set_ylabel(r'$\log_{10} \tilde{\rho}_p$', fontsize=13)      
      ax[0, columnindex].set_xlabel(r'$\log_{10} \tilde{f}_p$', fontsize=13 )
      if model == name_synthetic:
         ax[0, columnindex].set_title('ABCDE'[columnindex] + ') ' + description_param_to_title[model]+' ' + {'vary_N_pheno': r'(vary $n_p$)', 'vary_T': r'(vary $T$)'}[type_plot])
      else:
         ax[0, columnindex].set_title('ABCDE'[columnindex] + ') '+description_param_to_title[model])
      ####
      ax[1, columnindex].scatter(df_stats['covariance neighbours'].tolist(), df_stats['covariance random'].tolist(), color=color_list[i], s=5)
      max_cov = max([max_cov, max(df_stats['covariance neighbours'].tolist()), max(df_stats['covariance random'].tolist())])
      ax[1, columnindex].set_xlabel("covariance\nneighbours "+r" $g$ and $g'$", fontsize=13)
      ax[1, columnindex].set_ylabel('covariance\nnull model '+r'$g$ and $h$', fontsize=13)
      ###
      exceptions_corr_analysis = len([s for s, t in zip(df_stats['covariance neighbours'].tolist(), df_stats['covariance random'].tolist()) if s < t or s <= 0 or np.isnan(s) or np.isnan(t)])
      if exceptions_corr_analysis:
         exceptions_undefined = len([s for s, t in zip(df_stats['covariance neighbours'].tolist(), df_stats['covariance random'].tolist()) if np.isnan(s) or np.isnan(t)])
         exceptions_negative  = len([s for s, t in zip(df_stats['covariance neighbours'].tolist(), df_stats['covariance random'].tolist()) if not np.isnan(s) and s <= 0])
         exceptions_less_null  = len([s for s, t in zip(df_stats['covariance neighbours'].tolist(), df_stats['covariance random'].tolist()) if not np.isnan(s) and s < t and s >= 0 and not np.isnan(t)])
         print(description_parameters_kbT, 'exceptions genetic correlations - Pearson analysis', exceptions_corr_analysis, 'out of', len([N for N in ND_N_list if not np.isnan(N) and N > 0]), 'out of which undefined', exceptions_undefined, 'out of which less than zero', exceptions_negative, 'out of which less than null model', exceptions_less_null)
   if type_plot == 'vary_T':
      try:
         df_DGPmap = pd.read_csv('./data/GPmapproperties_'+ 'synthetic_normal_' + str(N_pheno) + model +'.csv')
         D_f_list, D_rho_list = zip(*[(N/K**L, rho) for N, rho in zip(df_DGPmap['neutral set size'].tolist(), df_DGPmap['rho'].tolist()) if not np.isnan(N) and N > 0])
         ax[0, columnindex].scatter(np.log10(D_f_list), np.log10(D_rho_list), color='grey', s=5, lw = 0, zorder=-3)
         legend_elements = [Line2D([0], [0], mfc='grey', ls='', marker='o', label=r'determin.', mew=0, ms=5),] + legend_elements
      except IOError:
         pass
   xlims, ylims = ax[0, columnindex].get_xlim(), ax[0, columnindex].get_ylim()
   lims = (1.05*min(xlims[0], ylims[0]), max(xlims[1], ylims[1])) # 1.05 times a negative is smaller than lowest data point
   assert min(lims) < min(xlims) and min(lims) < min(ylims)
   ax[0, columnindex].plot([-20, 0], [-20, 0], c='k', zorder=-4)
   ax[0, columnindex].set_xlim(lims)
   ax[0, columnindex].set_ylim(lims)
   abs_ylims = np.abs(lims)
   if max(abs_ylims) - min(abs_ylims) < 6:
      ax[0, columnindex].set_yticks([-1*i for i in range(int(min(abs_ylims)), 1 + int(max(abs_ylims)))])
      ax[0, columnindex].set_xticks([-1*i for i in range(int(min(abs_ylims)), 1 + int(max(abs_ylims)))])
   else:
      ax[0, columnindex].set_yticks([-1*i for i in np.arange(int(min(abs_ylims)), 1 + int(max(abs_ylims)), step=3)])
      ax[0, columnindex].set_xticks([-1*i for i in np.arange(int(min(abs_ylims)), 1 + int(max(abs_ylims)), step=3)])
   xlims, ylims = ax[1, columnindex].get_xlim(), ax[1, columnindex].get_ylim()
   lims = (min([xlims[0], ylims[0]]) - 0.0001, max_cov * 1.01) 
   ax[1, columnindex].plot([-1, 1], [-1, 1], c='k', zorder=-4)
   ax[1, columnindex].set_xlim(lims)
   ax[1, columnindex].set_ylim(lims)
   if  0.03 < max(lims) < 0.04:
      ticks = [0, 0.02, 0.04]
   elif 0.06 < max(lims):
      ticks = [0, 0.03, 0.06]
   elif 0.007 < max(lims) < 0.01:
      ticks = [0, 0.005, 0.01]
   elif 0.007 > max(lims):
      ticks = [0, 0.0025, 0.005]
   elif 0.02 < max(lims)< 0.03:
      ticks = [0, 0.015, 0.03]
   ax[1, columnindex].set_xticks(ticks)
   ax[1, columnindex].set_yticks(ticks)
   ax[0, columnindex].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.1), 
                         ncol=2, fancybox=False, columnspacing=0.5, handletextpad=0.2, borderaxespad=3)
f.tight_layout()
f.savefig('./plots/genetic_correlations_plot_detailed'+'_'.join([param.HP_filename, param.RNA_filename, polyomino_params, name_synthetic])+'.png', bbox_inches='tight', dpi=300)
plt.close('all')
################################################################################################
print('\n\nevolvability plot', flush=True)
###############################################################################################
for plot_norm in ['unnormalised', '']:
   f, ax = plt.subplots(ncols=5, nrows=2, figsize=(11, 5))
   for columnindex, (model, type_plot) in enumerate([('RNA12', 'all'), ('polyomino', 'all'), ('HP16_2Li', 'all'), (name_synthetic, 'vary_N_pheno'), (name_synthetic, 'vary_T')]):
      if model == name_synthetic:
         list_Npheno_andT, color_list, legend_elements = type_plot_vs_n_and_kbT_list[type_plot], model_vs_colors[type_plot], model_vs_legend_elements[type_plot]
      else:
         list_Npheno_andT, color_list, legend_elements = [('none', kbT) for kbT in type_map_vs_kbT_list[model]], model_vs_colors[model], model_vs_legend_elements[model]
      max_evolv = 0
      for i, (N_pheno, kbT) in enumerate(list_Npheno_andT):
         if model == 'polyomino':
            description_parameters_kbT = model + str(kbT) + polyomino_params
            K, L = int(kbT.split('_')[1]), int(kbT.split('_')[0]) * 4
         elif model == name_synthetic:
            description_parameters_kbT = 'synthetic_normal_' + str(N_pheno) + model + '_kbT' + str(kbT)
         else:
            description_parameters_kbT = model +'_kbT'+str(kbT)
         if model != 'polyomino':
            K, L = model_vs_KL[model]
         NDGrobustness_filename, NDGevolvability_filename = './data/NDGrobustness_'+description_parameters_kbT+'.npy', './data/NDGevolvability_'+description_parameters_kbT+'.npy'
         try:
            NDGrobustness =  np.load(NDGrobustness_filename)
            NDGevolvability = np.load(NDGevolvability_filename)
            df_NDGPmap = pd.read_csv('./data/NDGPmapproperties_'+description_parameters_kbT+'.csv')
         except IOError:
            print('not found', description_parameters_kbT)
            continue
         max_evolv = max(max_evolv, np.max(NDGevolvability))
         ##
         ax[0, columnindex].scatter(NDGrobustness.flatten(), NDGevolvability.flatten(), s=0.5, alpha=0.1, color=color_list[i])
         ax[0, columnindex].set_ylabel(r'$\tilde{\epsilon}_g$', fontsize=13) 
         ax[0, columnindex].set_xlabel( r'$\tilde{\rho}_g$', fontsize=13)
         for grho, gev in zip(NDGrobustness.flatten(), NDGevolvability.flatten()):
            assert gev <= (1-grho)* (K-1)*L + 0.0001
         if model == 'polyomino':
            xvalues = np.linspace(0.8 * np.min(NDGrobustness.flatten()), 1, 2)
            ax[0, columnindex].plot(xvalues, (1-xvalues)* (K-1)*L, color=color_list[i], ls=':', lw=0.7)
            #ax[0, columnindex].set_ylim(0, max_evolv*1.2)
         if model == name_synthetic:
            ax[0, columnindex].set_title('ABCDE'[columnindex] + ') ' + description_param_to_title[model]+' ' + {'vary_N_pheno': r'(vary $n_p$)', 'vary_T': r'(vary $T$)'}[type_plot])
         else:
            ax[0, columnindex].set_title('ABCDE'[columnindex] + ') '+description_param_to_title[model])
         #######
         ND_N_list_nonzero = [N for N in df_NDGPmap['neutral set size'].tolist() if not np.isnan(N) and N > 0]
         N_pheno = len(ND_N_list_nonzero)
         ND_rho_list = df_NDGPmap['rho'].tolist()
         ND_ev_list = df_NDGPmap['p_evolv'].tolist()
         print(description_parameters_kbT, 'min evolv', min(ND_ev_list)/(N_pheno-1))
         if plot_norm == '':
            ax[1, columnindex].scatter(ND_rho_list, np.divide(ND_ev_list, N_pheno - 1), color=color_list[i],  s=6, lw = 0)
         else:
            ax[1, columnindex].scatter(ND_rho_list, ND_ev_list, color=color_list[i],  s=6, lw = 0, alpha=0.4, zorder = 5-i)
            ax[1, columnindex].set_yscale('log')
         ax[1, columnindex].set_xlabel(r'$ \tilde{\rho}_p$', fontsize=13)      
         ax[1, columnindex].set_ylabel(r'$\tilde{\epsilon}_p/(n_p-1)$', fontsize=13 )
         if plot_norm == '':
            ax[1, columnindex].set_ylim(0, 1.08)
         ax[1, columnindex].set_xlim(-0.05, 1)
         #if stats.pearsonr(ND_rho_list, ND_ev_list)[0] < 0.7:
         print(description_parameters_kbT, 'corr rho-ev', stats.pearsonr(ND_rho_list, ND_ev_list)[0])
         assert min(ND_rho_list) >= 0 and max(ND_rho_list) <= 1 and  min(np.divide(ND_ev_list, N_pheno - 1)) >= 0 and max(np.divide(ND_ev_list, N_pheno - 1)) <= 1
      if type_plot == 'vary_T':
         try:
            df_DGPmap = pd.read_csv('./data/GPmapproperties_synthetic_normal_' + str(N_pheno) + model +'.csv')
            D_N_list_nonzero = [N for N in df_DGPmap['neutral set size'].tolist() if not np.isnan(N) and N > 0]
            N_pheno = len(D_N_list_nonzero)
            D_ev_list, D_rho_list = zip(*[(e/(N_pheno-1), rho) for e, rho in zip(df_DGPmap['p_evolv'].tolist(), df_DGPmap['rho'].tolist()) if not np.isnan(rho)])
            if plot_norm == '':
               ax[1, columnindex].scatter(D_rho_list, D_ev_list, color='grey', s=5, lw = 0, zorder=-3)
            legend_elements = [Line2D([0], [0], mfc='grey', ls='', marker='o', label=r'determin.', mew=0, ms=5),] + legend_elements
         except IOError:
            pass
      if model != 'polyomino':
         xvalues = np.array(ax[0, columnindex].get_xlim())
         ax[0, columnindex].plot(xvalues, (1-xvalues)* (K-1)*L, color='k', ls=':', lw=0.7)
         if model == 'RNA12':
            ax[0, columnindex].set_ylim(0, max_evolv*1.6)
         else:
            ax[0, columnindex].set_ylim(0, max_evolv*1.2)
         ax[0, columnindex].set_xlim(xvalues)
      ax[0, columnindex].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.1), 
                            ncol=2, fancybox=False, columnspacing=0.5, handletextpad=0.2, borderaxespad=3)
   f.tight_layout()
   f.savefig('./plots/evolv_plot'+plot_norm+'_'.join([param.HP_filename, param.RNA_filename, polyomino_params, name_synthetic])+'.png', bbox_inches='tight', dpi=300)
   plt.close('all')


################################################################################################
print('\n\nSappington/Mohanty plots', flush=True)
################################################################################################
f, ax = plt.subplots(ncols=5, nrows=2, figsize=(12, 5.6))
for columnindex, (model, type_plot) in enumerate([('RNA12', 'all'), ('polyomino', 'all'), ('HP16_2Li', 'all'), (name_synthetic, 'vary_N_pheno'), (name_synthetic, 'vary_T')]):
   if model == name_synthetic:
      list_Npheno_andT, color_list, legend_elements = type_plot_vs_n_and_kbT_list[type_plot], model_vs_colors[type_plot], model_vs_legend_elements[type_plot]
   else:
      list_Npheno_andT, color_list, legend_elements = [('none', kbT) for kbT in type_map_vs_kbT_list[model]], model_vs_colors[model], model_vs_legend_elements[model]
   for i, (N_pheno, kbT) in enumerate(list_Npheno_andT):
      if model == 'polyomino':
         description_parameters_kbT = model + str(kbT) + polyomino_params
         K, L = int(kbT.split('_')[1]), int(kbT.split('_')[0]) * 4
      elif model == name_synthetic:
         description_parameters_kbT = 'synthetic_normal_' + str(N_pheno) + model + '_kbT' + str(kbT)
      else:
         description_parameters_kbT = model +'_kbT'+str(kbT)
      if model != 'polyomino':
         K, L = model_vs_KL[model]
      NDGPmapdata_filename = './data/NDGPmapproperties_'+description_parameters_kbT+'.csv'
      try:
         df_NDGPmap = pd.read_csv(NDGPmapdata_filename)
         ND_N_list = df_NDGPmap['neutral set size'].tolist()
         ND_rho_list = df_NDGPmap['rho'].tolist()
         ND_entropy_list = df_NDGPmap['pheno entropy'].tolist()
      except IOError:
         print('not found', NDGPmapdata_filename)
         continue
      #####
      ax[0, columnindex].set_xlabel(r'$\log_{10} \tilde{f}_p$', fontsize=13)
      ax[0, columnindex].set_ylabel(r'$\log_{10} \tilde{\rho}_p$', fontsize=13) 
      rho, freq, entropy = zip(*[(ND_rho_list[i], N/K**L, ND_entropy_list[i]) for i, N in enumerate(ND_N_list) if N > 0 and ND_rho_list[i] > 0])
      ax[0, columnindex].scatter(np.log10(freq), np.log10(rho), color=color_list[i], s=5)
      xlims = ax[0, columnindex].get_xlim()
      logx_values = np.linspace(min(xlims), max(xlims), num=10**4)
      x_values = np.power(10, logx_values)
      upper_bound1 = [x*K**(L-1)/L if x < K**(1-L) else np.nan for x in x_values]
      upper_bound2 = [np.nan if x < K**(1-L) else 1 + np.log(x)/(L * np.log(K)) for x in x_values]
      if model == 'polyomino':
         color_upper_bound = color_list[i]
      else:
         color_upper_bound = 'grey'
      ax[0, columnindex].plot(np.log10(x_values), np.log10(upper_bound1), c=color_upper_bound, zorder=-1, lw=0.9)
      ax[0, columnindex].plot(np.log10(x_values), np.log10(upper_bound2), c=color_upper_bound, zorder=-1, lw=0.9)
      for x, y in zip(freq, rho):
         if x < K**(1-L):
            assert y < x*K**(L-1)/L
         else:
            assert y < 1 + np.log(x)/(L * np.log(K))
      ###
      rho, freq, entropy = zip(*[(ND_rho_list[i], N/K**L, ND_entropy_list[i]) for i, N in enumerate(ND_N_list)])
      predicted_robustness = [K**L*f*ent*np.exp(-1*ent)/(L * np.log(K)) for f, ent in zip(freq, entropy)]
      ax[1, columnindex].set_xlabel(r'$\tilde{\rho}_p$', fontsize=13)
      ax[1, columnindex].set_ylabel(r'predicted $\tilde{\rho}_p$' + '\nfrom '+r'$\tilde{S}_p$ and $\tilde{f}_p$', fontsize=13)    
      ax[1, columnindex].scatter(rho, predicted_robustness, color=color_list[i], s=5, alpha=0.9)  
      ax[1, columnindex].plot([0, 1], [0, 1], c='k', zorder=-1, lw=0.5)
      assert min(predicted_robustness) >= 0 and min(rho) >= 0 and max(predicted_robustness) <= 1 and max(rho) <= 1
      ax[1, columnindex].set_xlim(0, 1)
      ax[1, columnindex].set_ylim(0, 1)
   ax[0, columnindex].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.14), ncols=2)  
   for i in range(2):
      ax[i, columnindex].set_title(description_param_to_title[model])

f.tight_layout(rect=[0,0,1, 0.8])
f.savefig('./plots/Mohanty_analysis'+'_'.join([param.HP_filename, param.RNA_filename,  polyomino_params, name_synthetic])+'.png', bbox_inches='tight', dpi=250)
plt.close('all')


###############################################################################################
print('\n\nplot NC vs phenotypic quantities', flush=True)
###############################################################################################
thresholdNC = 0.2
f, ax = plt.subplots(ncols=5, nrows=2, figsize=(11, 4.7))
for columnindex, (model, type_plot) in enumerate([('RNA12', 'all'), ('polyomino', 'all'), ('HP16_2Li', 'all'), (name_synthetic, 'vary_N_pheno'), (name_synthetic, 'vary_T')]):
   if model == name_synthetic:
      list_Npheno_andT, color_list, legend_elements = type_plot_vs_n_and_kbT_list[type_plot], model_vs_colors[type_plot], model_vs_legend_elements[type_plot]
   else:
      list_Npheno_andT, color_list, legend_elements = [('none', kbT) for kbT in type_map_vs_kbT_list[model]], model_vs_colors[model], model_vs_legend_elements[model]
   for i, (N_pheno, kbT) in enumerate(list_Npheno_andT):
      if model == 'polyomino':
         description_parameters_kbT = model + str(kbT) + polyomino_params
         K, L = int(kbT.split('_')[1]), int(kbT.split('_')[0]) * 4
      elif model == name_synthetic:
         description_parameters_kbT = 'synthetic_normal_' + str(N_pheno) + model + '_kbT' + str(kbT)
      else:
         description_parameters_kbT = model +'_kbT'+str(kbT)
      if model != 'polyomino':
         K, L = model_vs_KL[model]
      try:
         df_NDGPmap = pd.read_csv('./data/NDGPmapproperties_'+description_parameters_kbT+'.csv')
         df_NDGPmap_NCs = pd.read_csv('./data/NDGPmappropertiesNCs_'+description_parameters_kbT+'_'+str(thresholdNC)+'.csv')
      except IOError:
         print('not found', description_parameters_kbT)
         continue
      N_pheno = len([N for N in df_NDGPmap['neutral set size'].tolist() if not np.isnan(N) and N > 0])
      ratio_NC_set_size = [x for x in np.divide(df_NDGPmap_NCs['neutral set size'].tolist(), df_NDGPmap['neutral set size'].tolist()) if not np.isnan(x)]
      ax[0, columnindex].hist(ratio_NC_set_size, alpha=0.5, color=color_list[i], density=False, bins=np.linspace(0, 1, 20), zorder=[i, 10 -i][columnindex%2])
      ax[0, columnindex].set_xlabel(r'$\tilde{f}_{NC}/\tilde{f}_{p}$', fontsize=13)  
      ax[0, columnindex].set_ylabel(r'number of phenotypes', fontsize=13)
      #
      ax[1, columnindex].scatter(df_NDGPmap_NCs['rho'].tolist(), np.divide(df_NDGPmap_NCs['p_evolv'].tolist(), N_pheno - 1), s=5, alpha=0.9, color=color_list[i])
      ax[1, columnindex].set_ylabel(r'$\tilde{\epsilon}_{NC}/(n_p -1)$', fontsize=13) 
      ax[1, columnindex].set_xlabel(r'$\tilde{\rho}_{NC}$', fontsize=13)
      if model == name_synthetic:
         ax[0, columnindex].set_title('ABCDE'[columnindex] + ') ' + description_param_to_title[model]+' ' + {'vary_N_pheno': r'(vary $n_p$)', 'vary_T': r'(vary $T$)'}[type_plot]+ '\ncutoff c='+str(thresholdNC))
      else:
         ax[0, columnindex].set_title('ABCDE'[columnindex] + ') '+description_param_to_title[model]+ '\ncutoff c='+str(thresholdNC))
   ax[0, columnindex].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.1), 
                      ncol=2, fancybox=False, columnspacing=0.5, handletextpad=0.2, borderaxespad=3)
f.tight_layout()
f.savefig('./plots/NC_analysis'+str(thresholdNC)+'_'.join([param.HP_filename, param.RNA_filename, polyomino_params, name_synthetic])+'.png', bbox_inches='tight', dpi=300)
plt.close('all')
###############################################################################################
print('\n\np2**nbp check in RNA', flush=True)
###############################################################################################
from functions import RNA_functions as RNA_functions

thresholdNC = 0.2
f, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
model = 'RNA12'
list_all_possible_structures_db = [s for s in RNA_functions.generate_all_allowed_dotbracket(L, allow_isolated_bps=True, filename='./data/allRNAstructuresRNA12.json') if '(' in s]
list_all_possible_structures = list(range(1, len(list_all_possible_structures_db) + 1))
structure_int_to_db = {i + 1: s for i, s in enumerate(list_all_possible_structures_db)}
structure_int_to_db[0] = '.' * L
db_to_structure_int = {s: i for i, s in structure_int_to_db.items()}
color_list, legend_elements = model_vs_colors[model], model_vs_legend_elements[model]
for i, kbT in enumerate(type_map_vs_kbT_list[model]):
   description_parameters_kbT = model +'_kbT'+str(kbT)
   K, L = model_vs_KL[model]
   try:
      df_NDGPmap = pd.read_csv('./data/NDGPmapproperties_'+description_parameters_kbT+'.csv')
      df_NDGPmap_NCs = pd.read_csv( './data/NDGPmappropertiesNCs_'+description_parameters_kbT+'_'+str(thresholdNC)+'.csv')
   except IOError:
      print('not found', description_parameters_kbT)
      continue
   ax[0].scatter(np.divide(df_NDGPmap_NCs['neutral set size'].tolist(), K**L), np.divide(df_NDGPmap['neutral set size'].tolist(), K**L), s=5, alpha=0.9, color=color_list[i])
   ax[0].set_ylabel(r'$\tilde{f}_p$') 
   ax[0].set_xlabel(r'$\tilde{f}_{NC}$')
   ###
   corrected_size = [N*2**structure_int_to_db[s].count('(') for N, s in zip(df_NDGPmap_NCs['neutral set size'].tolist(), df_NDGPmap_NCs['phenotype'].tolist())]
   ax[1].scatter(np.divide(corrected_size, K**L), np.divide(df_NDGPmap['neutral set size'].tolist(), K**L), s=4, alpha=0.9, color=color_list[i])
   ax[1].set_ylabel(r'$\tilde{f}_p$')  
   ax[1].set_xlabel(r'$\tilde{f}_{NC} \times 2^{bp}$')  
   
   for i in range(2):
      ax[i].set_xscale('log')
      ax[i].set_yscale('log')
      xlims = (min(ax[i].get_xlim()[0], ax[i].get_ylim()[0]), max(ax[i].get_xlim()[1], ax[i].get_ylim()[1]))
      ax[i].plot(xlims, xlims, color='k', ls=':', lw=0.7)
      ax[i].set_ylim(xlims[0], xlims[1])
      ax[i].set_xlim(xlims[0], xlims[1])
ax[0].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.1), 
                      ncol=2, fancybox=False, columnspacing=0.5, handletextpad=0.2, borderaxespad=3)
f.tight_layout()
f.savefig('./plots/RNA_NCs'+str(thresholdNC)+param.RNA_filename+'.png', bbox_inches='tight', dpi=300)
plt.close('all')
