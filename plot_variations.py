import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import isfile
from scipy import stats as stats
import pandas as pd
from matplotlib.lines import Line2D
import seaborn as sns
from copy import deepcopy
#import warnings
#warnings.filterwarnings("error")




def collect_stats(description_parameters_list, bias_plot_only_phenotypes_present_at_low_p=False):
   list_median_Pmfe, list_bias, list_genetic_corr, list_rho_evolv_corr, kbT_values_withdata, list_rho_span = [], [], [], [], [], []
   for description_parameters_kbT in description_parameters_list:
      try:
         df_NDGPmap = pd.read_csv('./data/NDGPmapproperties_'+description_parameters_kbT+'.csv')
         mfe_P_array = np.load('./data/mfe_P_array_'+description_parameters_kbT+'.npy')
         assert np.nanmin(mfe_P_array.flatten()) > 0 or type_plot == 'ReLu' #undefined only exists for ReLu (in the unlikely event that all G are positive)
         mfe_data = mfe_P_array.flatten()
         list_median_Pmfe.append((float(np.percentile(mfe_data, 25)), float(np.median(mfe_data)), float(np.percentile(mfe_data, 75))))
         kbT_values_withdata.append(float(description_parameters_kbT.split('T')[-1]))
         if not bias_plot_only_phenotypes_present_at_low_p:
            N_list = [N for N in df_NDGPmap['neutral set size'].tolist() if not np.isnan(N) and N > 0]
         else:
            if len(list_median_Pmfe) == 1:
               phenos_seen = set([p for p, N in zip(df_NDGPmap['phenotype'].tolist(), df_NDGPmap['neutral set size'].tolist()) if N > 0 and not np.isnan(N)])
            N_list = [N for p, N in zip(df_NDGPmap['phenotype'].tolist(), df_NDGPmap['neutral set size'].tolist()) if p in phenos_seen and not np.isnan(N) and N > 0]
         list_bias.append(np.log10(max(N_list)/min(N_list)))
         ratio_f_rho = [np.log10(rho/N) for N, rho in zip(np.divide(df_NDGPmap['neutral set size'].tolist(), K**L), df_NDGPmap['rho'].tolist()) if N > 0 and not np.isnan(N) and rho > 0]
         list_genetic_corr.append((float(np.percentile(ratio_f_rho, 25)), float(np.median(ratio_f_rho)), float(np.percentile(ratio_f_rho, 75))))
         rho_nonan, evo_nonan = zip(*[(r, e) for r, e in zip(df_NDGPmap['rho'].tolist(), df_NDGPmap['p_evolv'].tolist()) if not np.isnan(r) and not np.isnan(e)])
         if max(evo_nonan) - min(evo_nonan) >=  10**(-6):
            list_rho_evolv_corr.append(stats.pearsonr(rho_nonan, evo_nonan)[0])
         else:
            list_rho_evolv_corr.append(np.nan)
            #print(description_parameters_kbT, 'near constant input in rho-ev', max(rho_nonan) - min(rho_nonan), max(evo_nonan) - min(evo_nonan))
         list_rho_span.append(np.nanmax(df_NDGPmap['rho'].tolist()) - np.nanmin([rho for rho, N in zip(df_NDGPmap['rho'].tolist(), df_NDGPmap['neutral set size'].tolist()) if not np.isnan(N) and not N == 0]))
         if description_parameters_kbT == description_parameters_list[0]:
            print(description_parameters_kbT, 'min pheno-evolv', min([e for e in df_NDGPmap['p_evolv'].tolist()])/len([N for N in N_list if N > 0 and not np.isnan(N)]))
      except IOError:
         print('not found', description_parameters_kbT)
         continue
   return list_median_Pmfe, list_bias, list_genetic_corr, list_rho_evolv_corr, kbT_values_withdata, list_rho_span

def plot_NDmap_properties(plot_filename, description_parameters_list, label_list_lines):
   f, ax = plt.subplots(ncols=6, figsize=(13.5, 2.5), width_ratios=[1,]*5 +[0.5])
   color_list = [plt.get_cmap('viridis')(i) for i in np.linspace(0, 0.92, len(description_parameters_list))]
   for kbtindex, description_parameters_kbT in enumerate(description_parameters_list):
      try:
         df_NDGPmap = pd.read_csv('./data/NDGPmapproperties_'+description_parameters_kbT+'.csv')
         mfe_P_array = np.load('./data/mfe_P_array_'+description_parameters_kbT+'.npy') 
         NDGrobustness =  np.load('./data/NDGrobustness_'+description_parameters_kbT+'.npy')
         NDGevolvability = np.load('./data/NDGevolvability_'+description_parameters_kbT+'.npy')    
         ###
         freq_list_nonzero = [N/K**L for N in df_NDGPmap['neutral set size'] if not np.isnan(N) and N > 0]
         ax[0].scatter(np.log10(np.arange(1, len(freq_list_nonzero) + 1)), np.log10(sorted(freq_list_nonzero, reverse=True)), s=5, color =color_list[kbtindex])  
         ax[0].set_xlabel('log110 rank')
         ax[0].set_ylabel(r'log10 $\tilde{f}_p$')
         ####
         rho_list, freq_list = zip(*[(rho, N/K**L) for rho, N in zip(df_NDGPmap['rho'], df_NDGPmap['neutral set size']) if not np.isnan(N) and N > 0 and rho > 0 ])
         ax[1].scatter(np.log10(freq_list), np.log10(rho_list), s=5, color = color_list[kbtindex])  
         ax[1].set_xlabel(r'log10 $\tilde{f}_p$')
         ax[1].set_ylabel(r'log10 $\tilde{\rho}_p$')
         ####
         ax[2].scatter(NDGrobustness.flatten(), NDGevolvability.flatten(), s=5, color = color_list[kbtindex], alpha=0.4)  
         ax[2].set_xlabel(r'$\tilde{rho}_g$')
         ax[2].set_ylabel(r'$\tilde{\epsilon}_g$')
         for rho, ev in zip(NDGrobustness.flatten(), NDGevolvability.flatten()):
            assert ev < (1-rho) * (K-1)*L + 0.0001
         ####
         ax[3].scatter(df_NDGPmap['rho'], np.divide(df_NDGPmap['p_evolv'], len(freq_list_nonzero) -1), s=5, color = color_list[kbtindex])  
         ax[3].set_xlabel(r'$\tilde{rho}_p$')
         ax[3].set_ylabel(r'$\tilde{\epsilon}_p$ (norm.)')
         ####
         assert np.nanmin(mfe_P_array.flatten()) > -0.00001 and np.nanmax(mfe_P_array.flatten()) < 1.00001
         ax[4].hist(mfe_P_array.flatten(), color =color_list[kbtindex], bins=np.linspace(0, 1, 20), alpha=0.7)  
         ax[4].set_xlabel(r'highest-$P(p|g)$')
         ax[4].set_ylabel(r'freq')  
         ax[4].set_xlim(0, 1) 
      except IOError:
         print('not found', description_parameters_kbT)
         continue
   #####
   xlims, ylims = ax[1].get_xlim(), ax[1].get_ylim()
   lims = min(xlims[0], ylims[0]), max(xlims[1], ylims[1]),
   ax[1].plot(lims, lims, c='k', zorder=-2)
   #####
   xlims, ylims = np.array([0, 1]), ax[2].get_ylim()
   ax[2].plot(xlims, (1-xlims)*(K-1)*L, c='k', zorder=-2)
   ax[2].set_xlim(xlims)
   ax[2].set_ylim([0, ylims[1]])
   #####
   ax[-1].legend(handles=[Line2D([0], [0], mfc=color_list[i], ls='', marker='o', label=r'$k_b T$ = '+str(label), mew=0, ms=5) for i, label in enumerate(label_list_lines)])
   ax[-1].axis('off')
   f.tight_layout()
   f.savefig(plot_filename, bbox_inches='tight', dpi=300)


################################################################################################
################################################################################################


plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[cm]{sfmath}'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'cm'

K, L = 2, 15
type_plot_vs_label = {'linear': 'linear', 'inversesquared': 'inverse-squared', 'expsquared': 'Gaussian', 
                     'Boltzmann-like': 'Boltzmann-like', 'ReLu': 'ReLu', 'Softplus': 'Softplus', 
                     'lognormal': 'lognormal', 'uniform': 'uniform', 'normal': 'normal', 
                     'normalisednormal': 'normalised normal', 'binary': 'binary', 'offsetnormal': 'normal with offset',
                     2: 'K=2', 4: 'K=4'}
N_pheno = 100

################################################################################################
print('\n\nalternative functional forms', flush=True)
################################################################################################
for plot_correction in ['only_low_T_phenos_shown', '']:
   f, ax = plt.subplots(ncols=3, nrows=2, figsize=(6, 3.6))
   color_list, function_list = ['k'] + sns.color_palette("tab10", 10)[:5], ['Boltzmann-like', 'linear', 'inversesquared', 'expsquared', 'ReLu', 'Softplus'] 
   for columnindex, type_plot in enumerate(function_list):
      if type_plot == 'Boltzmann-like':
         description_parameters = 'synthetic_normal_' + str(N_pheno)+'_'+ str(L) + '_' + str(K) 
      else:
         description_parameters = 'synthetic_diffnonlinear_function_'+type_plot+'_normal_' + str(N_pheno)+'_'+ str(L) + '_' + str(K)
      description_parameters_list = [description_parameters + '_kbT' + str(kbT) for kbT in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2.5, 5]]
      if plot_correction == 'only_low_T_phenos_shown':
         bias_plot_only_phenotypes_present_at_low_p = True
      else:
         bias_plot_only_phenotypes_present_at_low_p = False
      list_median_Pmfe, list_bias, list_genetic_corr, list_rho_evolv_corr, kbT_values_withdata, list_rho_span = collect_stats(description_parameters_list, bias_plot_only_phenotypes_present_at_low_p=bias_plot_only_phenotypes_present_at_low_p)
      if len(list_median_Pmfe) == 0:
         continue 
      ax[0, 0].plot(kbT_values_withdata, list_bias, color = color_list[columnindex], marker='o', linewidth=1.2, markersize=2)
      lower_q2, median2, upper_q2 = zip(*list_genetic_corr)
      ax[0, 1].errorbar(kbT_values_withdata, median2, yerr=(np.array(median2) - np.array(lower_q2), np.array(upper_q2)- np.array(median2)), color = color_list[columnindex], marker='o', linewidth=1.2, markersize=2, capsize=2)
      ax[1, 0].plot(kbT_values_withdata, list_rho_evolv_corr, color = color_list[columnindex], marker='o', linewidth=1.2, markersize=2)
      ax[0, 2].plot(kbT_values_withdata, list_rho_span, color = color_list[columnindex], marker='o', linewidth=1.2, markersize=2)
      lower_q, median, upper_q = zip(*list_median_Pmfe)
      ax[1, 1].errorbar(kbT_values_withdata, median, yerr=(np.array(median) - np.array(lower_q), np.array(upper_q)- np.array(median)), color = color_list[columnindex], marker='o', linewidth=1.2, markersize=2, capsize=2)
      if plot_correction == '':
         print(type_plot, 'median Pmfe/high-T expectation', max(median)/(1/N_pheno))
         description_parameters_list_examples, labels_list_examples = zip(*[(description_parameters + '_kbT' + str(kbT), kbT) for kbT in [0.01, 0.1, 0.5, 1, 2.5]])
         plot_NDmap_properties('./plots/NDGPmapproperties_' + type_plot + '.png', description_parameters_list_examples, labels_list_examples)
   ax[0, 0].set_ylabel('phenotypic bias\n'+r'$\log_{10}$(max($\tilde f_p$)/min($\tilde f_p$))')
   ax[0, 1].set_ylabel('genetic correlations\n'+r'$\log_{10} (\tilde{\rho_p}/\tilde f_p)$')
   ax[1, 0].set_ylabel('correlation\n'+r'$\tilde{\rho_p}$ vs $\tilde{\epsilon_p}$')
   ax[0, 2].set_ylabel('pheno. rob. range\n'+r'$\tilde{\rho}_p^{max} - \tilde{\rho}_p^{min}$')
   ax[1, 1].plot(kbT_values_withdata, [1/N_pheno,] * len(kbT_values_withdata), c='grey', ls=':')
   ax[1, 1].set_ylabel('$P(p_1|g)$ for highest-\n'+r'prob. $p_1$ in ensemble')
   ax[0, 0].plot(kbT_values_withdata, [0,] * len(kbT_values_withdata), c='grey', ls=':')
   ax[0, 1].plot(kbT_values_withdata, [0,] * len(kbT_values_withdata), c='grey', ls=':')
   ax[0, 2].plot(kbT_values_withdata, [0,] * len(kbT_values_withdata), c='grey', ls=':')
   ax[1, 0].plot(kbT_values_withdata, [0,] * len(kbT_values_withdata), c='grey', ls=':')
   for i in range(5):
      ax[i//3, i%3].set_xlabel(r'stochasticity $T$')
      ax[i//3, i%3].set_xscale('log')
      ax[i//3, i%3].set_title('ABCDEFG'[i], loc='left')
   custom_lines = [Line2D([0], [0], mfc=color_list[i], ls='', marker='o', label=type_plot_vs_label[type_plot], mew=0, ms=5) for i, type_plot in enumerate(function_list)]
   custom_lines.append(Line2D([0], [0], mfc='grey', ls=':', marker=None, label=r'limit $T \to \infty$', mew=0, ms=5, c='grey') )
   ax[1, -1].legend(handles=custom_lines)
   ax[1, -1].axis('off')
   f.tight_layout()
   f.savefig('./plots/funcnormal'+plot_correction+'.png', bbox_inches='tight', dpi=250)
   plt.close('all')
################################################################################################
print('\n\nalternative distributions', flush=True)
################################################################################################
for plot_correction in ['only_low_T_phenos_shown', '']:
   f, ax = plt.subplots(ncols=3, nrows=2, figsize=(6, 3.6))
   color_list, dist_list = ['k'] + sns.color_palette("tab10", 10)[5:10], ['normal', 'lognormal', 'uniform', 'normalisednormal', 'binary', 'offsetnormal']
   for columnindex, type_plot in enumerate(dist_list):
      description_parameters = 'synthetic_'+type_plot+'_' + str(N_pheno)+'_'+ str(L) + '_' + str(K)
      description_parameters_list = [description_parameters + '_kbT' + str(kbT) for kbT in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2.5, 5]]
      if plot_correction == 'only_low_T_phenos_shown':
         bias_plot_only_phenotypes_present_at_low_p = True
      else:
         bias_plot_only_phenotypes_present_at_low_p = False
      list_median_Pmfe, list_bias, list_genetic_corr, list_rho_evolv_corr, kbT_values_withdata, list_rho_span = collect_stats(description_parameters_list, bias_plot_only_phenotypes_present_at_low_p=bias_plot_only_phenotypes_present_at_low_p)
      if len(list_median_Pmfe) == 0:
         continue 
      lower_q, median, upper_q = zip(*list_median_Pmfe)
      ax[1, 1].errorbar(kbT_values_withdata, median, yerr=(np.array(median) - np.array(lower_q), np.array(upper_q)- np.array(median)), color = color_list[columnindex], marker='o', linewidth=1.2, markersize=2, capsize=2)
      ax[0, 0].plot(kbT_values_withdata, list_bias, color = color_list[columnindex], marker='o', linewidth=1.2, markersize=2)
      lower_q2, median2, upper_q2 = zip(*list_genetic_corr)
      ax[0, 1].errorbar(kbT_values_withdata, median2, yerr=(np.array(median2) - np.array(lower_q2), np.array(upper_q2)- np.array(median2)), color = color_list[columnindex], marker='o', linewidth=1.2, markersize=2, capsize=2)
      ax[1, 0].plot(kbT_values_withdata, list_rho_evolv_corr, color = color_list[columnindex], marker='o', linewidth=1.2, markersize=2)
      ax[0, 2].plot(kbT_values_withdata, list_rho_span, color = color_list[columnindex], marker='o', linewidth=1.2, markersize=2)
      if plot_correction == '':
         print(type_plot, 'max rho span', max(list_rho_span), 'max bias', max(list_bias))
         description_parameters_list_examples, labels_list_examples = zip(*[(description_parameters + '_kbT' + str(kbT), kbT) for kbT in [0.01, 0.1, 0.5, 1, 2.5]])
         plot_NDmap_properties('./plots/NDGPmapproperties_' + type_plot + '.png', description_parameters_list_examples, labels_list_examples)

   ax[0, 0].set_ylabel('phenotypic bias\n'+r'$\log_{10}$(max($\tilde f_p$)/min($\tilde f_p$))')
   ax[0, 1].set_ylabel('genetic correlations\n'+r'$\log_{10} (\tilde{\rho_p}/\tilde f_p)$')
   ax[1, 0].set_ylabel('correlation\n'+r'$\tilde{\rho_p}$ vs $\tilde{\epsilon_p}$')
   ax[0, 2].set_ylabel('pheno. rob. range\n'+r'$\tilde{\rho}_p^{max} - \tilde{\rho}_p^{min}$')
   ax[1, 1].set_ylabel('$P(p_1|g)$ for highest-\n'+r'prob. $p_1$ in ensemble')
   ax[1, 1].plot(kbT_values_withdata, [1/N_pheno,] * len(kbT_values_withdata), c='grey', ls=':')
   ax[0, 2].plot(kbT_values_withdata, [0,] * len(kbT_values_withdata), c='grey', ls=':')
   ax[1, 0].plot(kbT_values_withdata, [0,] * len(kbT_values_withdata), c='grey', ls=':')
   ax[0, 0].plot(kbT_values_withdata, [0,] * len(kbT_values_withdata), c='grey', ls=':')
   ax[0, 1].plot(kbT_values_withdata, [0,] * len(kbT_values_withdata), c='grey', ls=':')
   for i in range(5):
      ax[i//3, i%3].set_xlabel(r'stochasticity $T$')
      ax[i//3, i%3].set_title('ABCDEFG'[i], loc='left') 
      ax[i//3, i%3].set_xscale('log')
   custom_lines = [Line2D([0], [0], mfc=color_list[i], ls='', marker='o', label=type_plot_vs_label[type_plot], mew=0, ms=5) for i, type_plot in enumerate(dist_list)]
   custom_lines.append(Line2D([0], [0], mfc='grey', ls=':', marker=None, label=r'limit $T \to \infty$', mew=0, ms=5, c='grey') )
   ax[1, -1].legend(handles=custom_lines)
   ax[1, -1].axis('off')
   f.tight_layout()
   f.savefig('./plots/distr'+plot_correction+'.png', bbox_inches='tight', dpi=250)
   plt.close('all')
################################################################################################
print('\n\nlinear functional forms with lognormal', flush=True)
################################################################################################
f, ax = plt.subplots(ncols=3, nrows=2, figsize=(7, 4.25))
color_list, function_list = ['k'] + sns.color_palette("tab10", 10)[:1]+ sns.color_palette("tab10", 10)[3:5], ['Boltzmann-like', 'linear', 'ReLu', 'Softplus'] 
for columnindex, type_plot in enumerate(function_list):
   bias_data, genetic_corr_data, rho_data, pmfe_data, corr_ev_data = [], [], [], [], []
   for distribution in ['normal', 'lognormal']:
      if type_plot == 'Boltzmann-like':
         description_parameters = 'synthetic_'+distribution+'_' + str(N_pheno)+'_'+ str(L) + '_' + str(K)
      else:
         description_parameters = 'synthetic_diffnonlinear_function_'+type_plot+'_'+distribution+'_' + str(N_pheno)+'_'+ str(L) + '_' + str(K) 
      description_parameters_list = [description_parameters + '_kbT' + str(kbT) for kbT in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2.5, 5]]
      list_median_Pmfe, list_bias, list_genetic_corr, list_rho_evolv_corr, kbT_values_withdata, list_rho_span = collect_stats(description_parameters_list)
      bias_data.append(deepcopy(list_bias))
      genetic_corr_data.append([c[1] for c in list_genetic_corr])
      pmfe_data.append([c[1] for c in list_median_Pmfe])
      rho_data.append(deepcopy(list_rho_span))
      corr_ev_data.append(deepcopy(list_rho_evolv_corr))
      ###
      description_parameters_list_examples, labels_list_examples = zip(*[(description_parameters + '_kbT' + str(kbT), kbT) for kbT in [0.01, 0.1, 0.5, 1, 2.5]])
      plot_NDmap_properties('./plots/NDGPmapproperties_' + type_plot + distribution + '.png', description_parameters_list_examples, labels_list_examples)


   ax[0, 0].scatter(bias_data[0], bias_data[1], color = color_list[columnindex], s=8, marker='x', linewidths=0.5)
   ax[0, 1].scatter(genetic_corr_data[0], genetic_corr_data[1], color = color_list[columnindex], s=8, marker='x', linewidths=0.5)
   ax[1, 0].scatter(corr_ev_data[0], corr_ev_data[1], color = color_list[columnindex], s=8, marker='x', linewidths=0.5)
   ax[0, 2].scatter(rho_data[0], rho_data[1], color = color_list[columnindex], s=8, marker='x', linewidths=0.5)
   ax[1, 1].scatter(pmfe_data[0], pmfe_data[1], color = color_list[columnindex], s=8, marker='x', linewidths=0.5)

ax[0, 0].set_ylabel('log-normal init.\n'+r'$\log_{10}$(max($\tilde f_p$)/min($\tilde f_p$))')
ax[0, 0].set_xlabel('normal init.\n'+r'$\log_{10}$(max($\tilde f_p$)/min($\tilde f_p$))')
ax[0, 1].set_ylabel('log-normal init.\n'+r'$\log_{10} (\tilde{\rho_p}/\tilde f_p)$')
ax[1, 0].set_ylabel('log-normal init.\n'+r'corr. $\tilde{\rho_p}$ vs $\tilde{\epsilon_p}$')
ax[0, 2].set_ylabel('log-normal init.\n'+r'$\tilde{\rho}_p^{max} - \tilde{\rho}_p^{min}$')
ax[1, 1].set_ylabel('log-normal init.\n'+'$P(p_1|g)$ for highest-\n'+r'prob. $p_1$ in ensemble')
ax[0, 1].set_xlabel('normal init.\n'+r'$\log_{10} (\tilde{\rho_p}/\tilde f_p)$')
ax[1, 0].set_xlabel('normal init.\n'+r'corr. $\tilde{\rho_p}$ vs $\tilde{\epsilon_p}$')
ax[0, 2].set_xlabel('normal init.\n'+r'$\tilde{\rho}_p^{max} - \tilde{\rho}_p^{min}$')
ax[1, 1].set_xlabel('normal init.\n'+'$P(p_1|g)$ for highest-\n'+r'prob. $p_1$ in ensemble')
for i in range(5):
   ax[i//3, i%3].set_title('ABCDEFG'[i], loc='left')
   xlims, ylims = ax[i//3, i%3].get_xlim(), ax[i//3, i%3].get_ylim()
   lim = (min(xlims[0], ylims[0]), max(xlims[1], ylims[1]))
   ax[i//3, i%3].set_xlim(lim)
   ax[i//3, i%3].set_ylim(lim)
   ax[i//3, i%3].plot(lim, lim, ls=':', c='grey')
custom_lines = [Line2D([0], [0], mfc=color_list[i], ls='none', marker='o', label=type_plot_vs_label[type_plot], mew=0, ms=5) for i, type_plot in enumerate(function_list)]
ax[1, -1].legend(handles=custom_lines)
ax[1, -1].axis('off')
f.tight_layout()
f.savefig('./plots/linear_lognormal.png', bbox_inches='tight', dpi=250)
plt.close('all')


################################################################################################
print('\n\nalphabet size', flush=True)
################################################################################################
f, ax = plt.subplots(ncols=3, nrows=2, figsize=(6, 3.6))
color_list = ['k'] + ['teal']
for columnindex, type_plot in enumerate([2, 4]):
   L, K = {2: 15, 4:8}[type_plot], type_plot
   description_parameters = 'synthetic_normal_' + str(N_pheno)+'_'+ str(L) + '_' + str(K)
   description_parameters_list = [description_parameters + '_kbT' + str(kbT) for kbT in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2.5, 5]]
   list_median_Pmfe, list_bias, list_genetic_corr, list_rho_evolv_corr, kbT_values_withdata, list_rho_span = collect_stats(description_parameters_list)
   if len(list_median_Pmfe) == 0:
      continue 
   ###
   description_parameters_list_examples, labels_list_examples = zip(*[(description_parameters + '_kbT' + str(kbT), kbT) for kbT in [0.01, 0.1, 0.5, 1, 2.5]])
   plot_NDmap_properties('./plots/NDGPmapproperties_K' + str(type_plot) + '.png', description_parameters_list_examples, labels_list_examples)

   
   ax[0, 0].plot(kbT_values_withdata, list_bias, color = color_list[columnindex], marker='o', linewidth=1.2, markersize=2)
   lower_q2, median2, upper_q2 = zip(*list_genetic_corr)
   ax[0, 1].errorbar(kbT_values_withdata, median2, yerr=(np.array(median2) - np.array(lower_q2), np.array(upper_q2)- np.array(median2)), color = color_list[columnindex], marker='o', linewidth=1.2, markersize=2, capsize=2)
   ax[1, 0].plot(kbT_values_withdata, list_rho_evolv_corr, color = color_list[columnindex], marker='o', linewidth=1.2, markersize=2)
   ax[0, 2].plot(kbT_values_withdata, list_rho_span, color = color_list[columnindex], marker='o', linewidth=1.2, markersize=2)
   lower_q, median, upper_q = zip(*list_median_Pmfe)  
   ax[1, 1].errorbar(kbT_values_withdata, median, yerr=(np.array(median) - np.array(lower_q), np.array(upper_q)- np.array(median)), color = color_list[columnindex], marker='o', linewidth=1.2, markersize=2, capsize=2)

ax[0, 0].set_ylabel('phenotypic bias\n'+r'$\log_{10}$(max($\tilde f_p$)/min($\tilde f_p$))')
ax[0, 1].set_ylabel('genetic correlations\n'+r'$\log_{10} (\tilde{\rho_p}/\tilde f_p)$')
ax[1, 0].set_ylabel('correlation\n'+r'$\tilde{\rho_p}$ vs $\tilde{\epsilon_p}$')
ax[0, 2].set_ylabel('pheno. rob. range\n'+r'$\tilde{\rho}_p^{max} - \tilde{\rho}_p^{min}$')
ax[1, 1].plot(kbT_values_withdata, [1/N_pheno,] * len(kbT_values_withdata), c='grey', ls=':')
ax[1, 1].set_ylabel('$P(p_1|g)$ for highest-\n'+r'prob. $p_1$ in ensemble')
ax[0, 0].plot(kbT_values_withdata, [0,] * len(kbT_values_withdata), c='grey', ls=':')
ax[0, 1].plot(kbT_values_withdata, [0,] * len(kbT_values_withdata), c='grey', ls=':')
ax[0, 2].plot(kbT_values_withdata, [0,] * len(kbT_values_withdata), c='grey', ls=':')
ax[1, 0].plot(kbT_values_withdata, [0,] * len(kbT_values_withdata), c='grey', ls=':')
for i in range(5):
   ax[i//3, i%3].set_xlabel(r'stochasticity $T$')
   ax[i//3, i%3].set_xscale('log')
   ax[i//3, i%3].set_title('ABCDEFG'[i], loc='left')
custom_lines = [Line2D([0], [0], mfc=color_list[i], ls='', marker='o', label=type_plot_vs_label[type_plot], mew=0, ms=5) for i, type_plot in enumerate([2, 4])]
custom_lines.append(Line2D([0], [0], mfc='grey', ls=':', marker=None, label=r'limit $T \to \infty$', mew=0, ms=5, c='grey') )
ax[1, -1].legend(handles=custom_lines)
ax[1, -1].axis('off')
f.tight_layout()
f.savefig('./plots/alphabet_size.png', bbox_inches='tight', dpi=250)
plt.close('all')

################################################################################################
print('\n\ndeterminants of neutral set size with constant offset', flush=True)
################################################################################################
K, L = 2, 15
f, ax = plt.subplots(ncols=3, figsize=(9, 2))
type_plot = 'offsetnormal'
for columnindex, kbT in enumerate([0.1, 0.5, 2.5]):
   description_parameters_kbT = 'synthetic_'+type_plot+'_' + str(N_pheno)+'_'+ str(L) + '_' + str(K) + '_kbT' + str(kbT)
   df_NDGPmap = pd.read_csv('./data/NDGPmapproperties_'+description_parameters_kbT+'.csv')   
   ND_N_list = df_NDGPmap['neutral set size'].tolist()
   structure_vs_structure_vect_df = pd.read_csv('data/parameter_vectors_'+'synthetic_'+type_plot+'_' + str(N_pheno)+'_'+ str(L) + '_' + str(K)+'.csv')
   structure_vs_structure_vect = {rowi['ph']: [float(x) for x in rowi['structure vector'].split('_')] for i, rowi in structure_vs_structure_vect_df.iterrows()}
   last_vector_element = [structure_vs_structure_vect[int(s)][-1] for s in df_NDGPmap['phenotype'].tolist()]
   sc = ax[columnindex].scatter(df_NDGPmap['length phenotypic vector'].tolist(), last_vector_element, c=np.log10(np.divide(ND_N_list, K**L)), s=4, lw = 0)
   cb = f.colorbar(sc, ax=ax[columnindex])
   cb.set_label(r'$\tilde{f}_p$')
   ax[columnindex].set_xlabel('vector norm')
   ax[columnindex].set_ylabel('offset')
   ax[columnindex].set_title('kbT'+str(kbT))
f.tight_layout()
f.savefig('./plots/const_off'+'.png', bbox_inches='tight', dpi=250)
plt.close('all')





