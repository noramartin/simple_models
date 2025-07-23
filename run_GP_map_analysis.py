#!/usr/bin/env python3

import sys
print(sys.version)
import numpy as np
from functions import synthetic_model_functions as synthetic
from os.path import isfile
from scipy import stats
from functions import RNA_functions as RNA_functions
from functions import HPfunctions_numba as HP_functions
from functions import polyomino_functions as polyomino_functions
import pandas as pd
from math import isqrt
from functools import partial
import parameters as param
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functions.general_functions import *
from collections import Counter



print(sys.argv[1:])
###############################################################################################
###############################################################################################
print('parameters', flush=True)
###############################################################################################
###############################################################################################
type_map = sys.argv[1]
filepath = './'
###############################################################################################
###############################################################################################
print('check parameters', flush=True)
###############################################################################################
###############################################################################################
if type_map.startswith('RNA'):
   L, K = param.RNA_L, param.RNA_K
   description_parameters = param.RNA_filename
   list_all_possible_structures_db = [s for s in RNA_functions.generate_all_allowed_dotbracket(L, allow_isolated_bps=True, filename=filepath + 'data/allRNAstructures' + description_parameters +'.json') if '(' in s]
   list_all_possible_structures = list(range(1, len(list_all_possible_structures_db) + 1))
   structure_int_to_db = {i + 1: s for i, s in enumerate(list_all_possible_structures_db)}
   structure_int_to_db[0] = '.' * L
   db_to_structure_int = {s: i for i, s in structure_int_to_db.items()}
   print(db_to_structure_int)
###############################################################################################
elif type_map.startswith('HP') or type_map.startswith('ncHP'):
   L, K, potential = param.HP_L, param.HP_K, param.HP_potential
   assert K == 2 or (potential == 'hHYX' and K == 4)
   description_parameters = param.HP_filename

   contact_map_list, updown_list, number_s_with_cm_list =  HP_functions.enumerate_all_structures_or_from_file(L, 
                                                                                                              contact_map_list_filename=filepath + 'data/allHPstructures' + '_'.join(description_parameters.split('_')[:-1]) +'.csv', 
                                                                                                              compact=param.HP_iscompact)
   structure_int_to_cm = {i + 1: s for i, s in enumerate(contact_map_list)}
   structure_int_to_ud = {i + 1: s for i, s in enumerate(updown_list)}
   assert  max(number_s_with_cm_list) == min(number_s_with_cm_list) #if there are degeneracies, temperature matters even for D GP maps
   list_all_possible_structures = sorted(list(structure_int_to_cm.keys()))
   ###
   number_s_with_cm_list = np.array(number_s_with_cm_list)
   contact_maps_as_single_array = HP_functions.contact_maps_list_to_single_array(L, contact_map_list, filepath + 'data/allHPstructures' + '_'.join(description_parameters.split('_')[:-1]) +'_singlenpy.npy')
###############################################################################################
elif type_map.startswith('polyomino'):
   description_parameters = param.polyomino_filename
   L, K = 4* param.polyomino_ntiles, param.polyomino_K
   print('K, L', K, L)
   #### genotype to assembly graph
   params_assembly_graph = str(param.polyomino_ntiles) + '_' + str(param.polyomino_K)
   if param.polyomino_seeded_assembly:
      params_assembly_graph =params_assembly_graph + 'seeded'
   genotype_to_assembly_graph_filename = filepath + 'data/genotype_to_assembly_graph_'+ params_assembly_graph +'.npy'
   if not isfile(genotype_to_assembly_graph_filename):
      genotype_to_assembly_graph, counter =  np.zeros((K,)*L, dtype='uint32'), 0
      assembly_graph_list, assembly_graph_characteristics = [], {}
      for g, zero in np.ndenumerate(genotype_to_assembly_graph):
         counter += 1
         if zero < 0.1: #not already provided info
            g_min = polyomino_functions.construct_simplest_genotype(g, max_c=K-1)
            if genotype_to_assembly_graph[g_min] > 0.1:
               genotype_to_assembly_graph[g] = genotype_to_assembly_graph[g_min]
            else:
               assembly_graph_index, assembly_graph_list, assembly_graph_characteristics =  polyomino_functions.find_assembly_graph(g, assembly_graph_list, assembly_graph_characteristics, seeded_assembly=param.polyomino_seeded_assembly)
               genotype_to_assembly_graph[g] = assembly_graph_index
               assert assembly_graph_index < 2**31
               list_of_genotypes_with_identical_assembly_graph = [tuple(g2[:]) for g2 in polyomino_functions.cyclic_permutations(np.array(g))]
               list_of_genotypes_with_identical_assembly_graph.append(tuple(g_min[:]))
               for g2 in list_of_genotypes_with_identical_assembly_graph:
                  assert len(g2) == L
                  assert genotype_to_assembly_graph[g2] < 0.1 or genotype_to_assembly_graph[g2] == assembly_graph_index
                  genotype_to_assembly_graph[g2] = assembly_graph_index
         if counter % (5*10**5) == 0:
            print('finished', counter/K**L*100, '%', 'number assembly graphs', len(assembly_graph_list), flush=True)
      np.save(genotype_to_assembly_graph_filename, genotype_to_assembly_graph, allow_pickle=False)
   else:
      genotype_to_assembly_graph = np.load(genotype_to_assembly_graph_filename)   
   list_assembly_graphs = np.arange(np.min(genotype_to_assembly_graph), np.max(genotype_to_assembly_graph) + 1)
   assert min(list_assembly_graphs) == 1 and len(list_assembly_graphs) == len(np.unique(genotype_to_assembly_graph))
   print('number of unique assembly graphs', len(list_assembly_graphs))
   #### assembly graph to phenotype probabilities
   assembly_graph_no_vs_pheno_ensemble_filename = filepath + 'data/assembly_graph_no_vs_pheno_ensemble_'+description_parameters+'.csv'
   pheno_vs_integer_filename = filepath + 'data/pheno_vs_integer_'+description_parameters+'.csv'
   try:
      assembly_graph_no_vs_pheno_ensemble = load_assembly_graph_no_vs_pheno_ensemble(assembly_graph_no_vs_pheno_ensemble_filename)
      pheno_vs_num_df = pd.read_csv(pheno_vs_integer_filename)
      pheno_vs_num = {row['phenotype']: row['number'] for i, row in pheno_vs_num_df.iterrows()}
   except IOError:
      assembly_graph_no_vs_pheno_ensemble, pheno_vs_num = {}, {0: 0}
      assembly_graph_int_to_ensemble_withparams = partial(polyomino_functions.assembly_graph_int_to_ensemble, genotype_to_assembly_graph=genotype_to_assembly_graph, 
                                                          n_runs=param.polyomino_n_runs, seeded_assembly=param.polyomino_seeded_assembly, threshold=param.polyomino_threshold)
      for assembly_graph in list_assembly_graphs:
         ensemble = assembly_graph_int_to_ensemble_withparams(assembly_graph)
         assert len(ensemble) > 0 and abs(sum(ensemble.values()) - 1) < 0.1/param.polyomino_n_runs and min(ensemble.values()) > 0.9/param.polyomino_n_runs
         for p in ensemble:
            if p not in pheno_vs_num.keys():
               pheno_vs_num[p] = max(list(pheno_vs_num.values())) + 1
         assembly_graph_no_vs_pheno_ensemble[assembly_graph]  = {pheno_vs_num[p]: P for p, P in ensemble.items()}
      ph_list_to_save = list(pheno_vs_num.keys())
      pd.DataFrame.from_dict({'phenotype': ph_list_to_save, 'number': list([pheno_vs_num[ph] for ph in ph_list_to_save])}).to_csv(pheno_vs_integer_filename)
      save_assembly_graph_no_vs_pheno_ensemble(assembly_graph_no_vs_pheno_ensemble_filename, assembly_graph_no_vs_pheno_ensemble)
   list_all_possible_structures = [e for e in pheno_vs_num.values() if e > 0]
###############################################################################################
elif type_map.startswith('synthetic'):
   K, L = param.synthetic_K, param.synthetic_L
   description_parameters = type_map + param.synthetic_filename
   number_phenos = int(type_map.split('_')[-1])
   distribution = type_map.split('_')[-2]
   filename_vectors = filepath + 'data/parameter_vectors_'+description_parameters+'.csv'
   if not isfile(filename_vectors):
      structure_vs_structure_vect = {s: synthetic.int_to_vector_structure_random(L * (K - 1), distribution=distribution) for s in range(1, number_phenos + 1)}
      list_all_possible_structures = [k for k in structure_vs_structure_vect.keys()]
      structure_vs_structure_vect_df = pd.DataFrame.from_dict({'ph': list_all_possible_structures,
                                                               'structure vector': ['_'.join([str(x) for x in structure_vs_structure_vect[p]]) for p in list_all_possible_structures]}).to_csv(filename_vectors)
   else:
      structure_vs_structure_vect_df = pd.read_csv(filename_vectors)
      structure_vs_structure_vect = {rowi['ph']: [float(x) for x in rowi['structure vector'].split('_')] for i, rowi in structure_vs_structure_vect_df.iterrows()}
      list_all_possible_structures = [k for k in structure_vs_structure_vect.keys()]
   if 'nonlinear_function' in type_map:
      different_nonlinear_function = True 
      name_different_function = type_map.split('_')[-3]
   else:
      different_nonlinear_function = False
      name_different_function = 'exponential'
   
###############################################################################################
else:
   raise RuntimeError('map does not exist')
###############################################################################################
assert max(list_all_possible_structures) < 2**31 and L < 32 and len(list_all_possible_structures) == max(list_all_possible_structures) == len(set(list_all_possible_structures))
assert 1 <= min(list_all_possible_structures)
print('list_all_possible_structures', len(list_all_possible_structures))
###############################################################################################
###############################################################################################
print('save GP map', flush=True)
###############################################################################################
if description_parameters.startswith('polyomino') and param.polyomino_traditionalGPmap:
   description_parameters += 'traditional_DGPmap'
GPmap_filename = filepath + 'data/GPmap_'+description_parameters+'.npy'
if not isfile(GPmap_filename):
   GPmap, counter =  np.zeros((K,)*L, dtype='uint32'), 0
   for g, zero in np.ndenumerate(GPmap):
      counter += 1
      if description_parameters.startswith('RNA'):
         mfe_structures = RNA_functions.get_all_mfe_structures_seq_str(g, db_to_int=db_to_structure_int)
      elif description_parameters.startswith('HP') or description_parameters.startswith('ncHP'):
         mfe_structures = HP_functions.find_mfe(g, contact_maps_as_single_array, number_s_with_cm_list, potential=potential, kbT=np.nan)
      elif description_parameters.startswith('polyomino'):
         ensemble = assembly_graph_no_vs_pheno_ensemble[genotype_to_assembly_graph[g]]
         if param.polyomino_traditionalGPmap: #only count ND assemblies
            if len(ensemble) == 1:
               mfe_structures = [p for p, f in ensemble.items()]
            else:
               mfe_structures = [0,]
         else:
            max_f = max(list(ensemble.values()))
            mfe_structures  =  [p for p, f in ensemble.items() if abs(f- max_f) < 0.5/n_runs]
      else:
         mfe_structures = synthetic.find_structures_that_minimise_G(g, structure_vs_structure_vect, resolution=10**(-4), K=K)
      if len(mfe_structures) == 1 and mfe_structures[0] >= 0:
         GPmap[g] = mfe_structures[0]
      else:
         GPmap[g] = 0
      if counter % 10**5 == 0:
         print('finished', counter/K**L*100, '%', flush=True)
   np.save(GPmap_filename, GPmap, allow_pickle=False)
else:
   GPmap = np.load(GPmap_filename)
if type_map.startswith('synthetic'):
   print('number of genotypes with undefined pheno', len(np.where(GPmap == 0)))
phenos_Dmap = [p for p in np.unique(GPmap) if p > 0]
print('number of unique phenos in D GP map', len(phenos_Dmap), np.unique(GPmap))
###############################################################################################
###############################################################################################
print('GP map analysis', flush=True)
###############################################################################################
###############################################################################################
GPmapdata_filename, Grobustness_filename, Gevolvability_filename = filepath + 'data/GPmapproperties_'+description_parameters+'.csv', filepath + 'data/Grobustness_'+description_parameters+'.npy', filepath + 'data/Gevolvability_'+description_parameters+'.npy'
if not isfile(GPmapdata_filename) or not isfile(Grobustness_filename) or not isfile(Gevolvability_filename):
   ph_vs_f, ph_vs_Prho, ph_vs_Pevolv = synthetic.get_deterministic_Prho_Pevolv(GPmap, structure_invalid_test = synthetic.isundefined_struct_int)
   N_list = [ph_vs_f[s] if s in ph_vs_f else np.nan for s in list_all_possible_structures]
   rho_list = [ph_vs_Prho[s] if s in ph_vs_f else np.nan for s in list_all_possible_structures]
   Pevolv_list = [ph_vs_Pevolv[s] if s in ph_vs_f else np.nan for s in list_all_possible_structures]
   data_dmap = {'phenotype': list_all_possible_structures, 'neutral set size': N_list, 'rho': rho_list, 'p_evolv': Pevolv_list}
   if type_map.startswith('synthetic'):
      data_dmap['sum over phenotypic vector'] = [sum(structure_vs_structure_vect[s]) for s in list_all_possible_structures]
      mean_no_phenos, std_no_phenos = synthetic.shape_space_covering(10**2, GPmap, structure_invalid_test= synthetic.isundefined_struct_int)
      df_shape_space = pd.DataFrame.from_dict({'mean # pheno': mean_no_phenos, 'std # phenos': std_no_phenos, 'dist': np.arange(1, len(mean_no_phenos) + 1)})
      df_shape_space.to_csv(filepath + 'data/shape_space_covering'+description_parameters+'.csv')
      ###
      ph_by_rank = sorted(ph_vs_f.keys(), key=ph_vs_f.get, reverse=True)
      phenos_for_versatility_analysis = [ph_by_rank[i] for i in [3, len(ph_by_rank)//4, len(ph_by_rank)//2]]
      data_versatility = {'pos': np.arange(1, L+1)}
      for ph in phenos_for_versatility_analysis:
         mean_v, std_v, NC_size, grho_list, no_neutral_vs_prevalence = synthetic.get_sequence_constraints(ph, GPmap)
         data_versatility['mean v  - rank '+str(len([n for n in N_list if n > ph_vs_f[ph]])) + ' ' + str(NC_size)] = list(mean_v)
         data_versatility['std v - rank '+str(len([n for n in N_list if n > ph_vs_f[ph]]))+ ' ' + str(NC_size)] = list(std_v) 
         for i in range(K):
            data_versatility[str(i) + 'neutral prevelance - rank '+str(len([n for n in N_list if n > ph_vs_f[ph]]))+ ' ' + str(NC_size)] = list(no_neutral_vs_prevalence[i]) 
         print('number of neutral neighbours', Counter(grho_list))
      pd.DataFrame.from_dict(data_versatility).to_csv(filepath + 'data/DGPmap_versatility'+description_parameters+'.csv')
      no_NCs_list = [np.max(synthetic.find_NCs(structure_int, GPmap)) for structure_int in ph_by_rank]
      pd.DataFrame.from_dict({'ph': ph_by_rank, '# NCs': no_NCs_list}).to_csv(filepath + 'data/DGPmap_nNCs'+description_parameters+'.csv')
   df_GPmap = pd.DataFrame.from_dict(data_dmap)
   df_GPmap.to_csv(GPmapdata_filename)
   Grobustness, Gevolvability = synthetic.get_Grobustness_Gevolvability(GPmap, structure_invalid_test=synthetic.isundefined_struct_int)
   np.save(Grobustness_filename, Grobustness, allow_pickle=False)
   np.save(Gevolvability_filename, Gevolvability, allow_pickle=False)
if 'traditional_DGPmap' in description_parameters:
   description_parameters = description_parameters.replace('traditional_DGPmap', '')
###############################################################################################
###############################################################################################
print('get ND GP map', flush=True)
###############################################################################################
###############################################################################################
if description_parameters.startswith('RNA'):
   assert '.' * L in db_to_structure_int.keys()
   kbT_list = [30, 37, 45]
elif description_parameters.startswith('polyomino'):
   kbT_list = ['default']
elif description_parameters.startswith('HP') or description_parameters.startswith('ncHP') :
   kbT_list = [0.001, 0.25, 0.5, 1]
elif description_parameters.startswith('synthetic'):
   kbT_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2.5, 5, 'mean_energygap']
for kbT in kbT_list:
   print(kbT, 'kbt')
   if kbT == 'default':
      description_parameters_kbT = description_parameters
   else:
      description_parameters_kbT = description_parameters +'_kbT'+str(kbT)
   if kbT == 'mean_energygap':
      assert type_map.startswith('synthetic')
      kbT = np.mean([synthetic.find_energygap(g, structure_vs_structure_vect, K=K) for g, P in np.ndenumerate(GPmap)])
   filename_Boltzmann_array = filepath + 'data/Parray'+description_parameters_kbT+'.npy'
   if isfile(filename_Boltzmann_array):
      P_array = np.load(filename_Boltzmann_array)
   elif not type_map.startswith('polyomino'):
      P_array, counter = np.zeros(tuple([K,] * L + [1 + len(list_all_possible_structures)]), dtype=np.float32), 0
      for g, structure_int in np.ndenumerate(GPmap):
         if description_parameters.startswith('synthetic'):
            assert name_different_function == 'exponential' or different_nonlinear_function
            Boltz_dist = synthetic.get_phenotype_ensemble(g, structure_vs_structure_vect, kbT = kbT, name_function=name_different_function, K=K)
         elif description_parameters_kbT.startswith('ncHP') or description_parameters_kbT.startswith('HP'):
            Boltz_dist = HP_functions.HPget_Boltzmann_freq(g, contact_maps_as_single_array=contact_maps_as_single_array, number_s_with_cm_list=number_s_with_cm_list, kbT=kbT, potential=param.HP_potential)
         elif description_parameters.startswith('RNA'):
            Boltz_dist = RNA_functions.get_Boltzmann_ensemble (g, db_to_structure_int=db_to_structure_int, temp=kbT)
         elif description_parameters.startswith('polyomino'):
            Boltz_dist = {ph: f for ph, f in assembly_graph_no_vs_pheno_ensemble[genotype_to_assembly_graph[g]].items()}
         assert abs(sum(Boltz_dist.values()) - 1) < 0.05
         for s in list_all_possible_structures + [0,]:
            try:
               P_array[tuple([x for x in g] + [s])] = Boltz_dist[s]
            except KeyError:
               pass
         del Boltz_dist
         counter += 1
         if counter % (5*10**4) == 0:
            print('finished', counter/K**L*100, '%', flush=True)
      if description_parameters.startswith('synthetic'): ### all-zero genotype will give artefacts - set to undefined
         P_array[tuple([0 for x in g] + [0])] = 1 
         for s in list_all_possible_structures:
            P_array[tuple([0 for x in g] + [s])] = 0
      np.save(filename_Boltzmann_array, P_array, allow_pickle=False)
   if type_map.startswith('polyomino'):
      assembly_graph_no_vs_pheno_ensemble_list = {a: [ensemble[i] if i in ensemble else 0 for i in range(1+max(list_all_possible_structures))] for a, ensemble in assembly_graph_no_vs_pheno_ensemble.items()}
      get_Boltzmann_ensemble_givenarray_list = partial(synthetic.get_Boltzmann_ensemble_list_polyomino, g_vs_assembly_graph=genotype_to_assembly_graph, assembly_graph_vs_ensemble_list=assembly_graph_no_vs_pheno_ensemble_list)
      def get_Boltzmann_ensemble_givenarray(g):
         return assembly_graph_no_vs_pheno_ensemble[genotype_to_assembly_graph[g]]
   else:
      get_Boltzmann_ensemble_givenarray_list = partial(synthetic.get_Boltzmann_ensemble_list, P_array = P_array)
      get_Boltzmann_ensemble_givenarray = partial(synthetic.get_Boltzmann_ensemble, P_array = P_array)

   ###############################################################################################
   print('analyse ND GP map', flush=True)
   ###############################################################################################
   NDGPmapdata_filename, NDGrobustness_filename, NDGevolvability_filename = filepath + 'data/NDGPmapproperties_'+description_parameters_kbT+'.csv', filepath + 'data/NDGrobustness_'+description_parameters_kbT+'.npy', filepath + 'data/NDGevolvability_'+description_parameters_kbT+'.npy'
   if not (isfile(NDGPmapdata_filename) and isfile(NDGrobustness_filename) and isfile(NDGevolvability_filename)):
      NDph_vs_f, NDph_vs_rho, NDPevolv_dict, NDPentropy, NDGrobustness, NDGevolvability = synthetic.ND_GPmapproperties(list_all_possible_structures, K, L, get_Boltzmann_ensemble_givenarray_list, structure_invalid_test=synthetic.isundefined_struct_int)
      NDPevolv_list = [NDPevolv_dict[s] for s in list_all_possible_structures]
      ND_N_list = [NDph_vs_f[s] for s in list_all_possible_structures]
      ND_rho_list = [NDph_vs_rho[s] for s in list_all_possible_structures]
      NDPentropy_list = [NDPentropy[s] for s in list_all_possible_structures]

      df_NDGPmap = pd.DataFrame.from_dict({'phenotype': list_all_possible_structures, 'neutral set size': ND_N_list, 'rho': ND_rho_list, 'p_evolv': NDPevolv_list, 'pheno entropy': NDPentropy_list})
      if type_map.startswith('synthetic'):
         df_NDGPmap['sum over phenotypic vector'] = [sum(structure_vs_structure_vect[s]) for s in list_all_possible_structures] 
      df_NDGPmap.to_csv(NDGPmapdata_filename)
      np.save(NDGrobustness_filename, NDGrobustness, allow_pickle=False)
      np.save(NDGevolvability_filename, NDGevolvability, allow_pickle=False)
   else:
      df_NDGPmap = pd.read_csv(NDGPmapdata_filename)
      NDph_vs_f = {p:f for p, f in zip(df_NDGPmap['phenotype'].tolist(), df_NDGPmap['neutral set size'].tolist())}
      NDph_vs_rho = {p:f for p, f in zip(df_NDGPmap['phenotype'].tolist(), df_NDGPmap['rho'].tolist())}
      NDPevolv_dict = {p:f for p, f in zip(df_NDGPmap['phenotype'].tolist(), df_NDGPmap['p_evolv'].tolist())}


   ###############################################################################################
   print('extract highest two probabilities for each geno', flush=True)
   ###############################################################################################
   if type_map.startswith('synthetic') and not isfile(filepath + 'data/mfe_P_array_'+description_parameters_kbT+'.npy'):
      mfe_P_array = synthetic.get_prob_lowest_G(K, L, get_Boltzmann_ensemble_givenarray)
      np.save(filepath + 'data/mfe_P_array_'+description_parameters_kbT+'.npy', mfe_P_array, allow_pickle=False)
   if type_map.startswith('synthetic') and not isfile(filepath + 'data/first_suboptimal_P_array_'+description_parameters_kbT+'.npy'):
      first_suboptimal_P_array = synthetic.get_prob_lowest_G(K, L, get_Boltzmann_ensemble_givenarray, type_prob= 'first_suboptimal')
      np.save(filepath + 'data/first_suboptimal_P_array_'+description_parameters_kbT+'.npy', first_suboptimal_P_array, allow_pickle=False)

   ###############################################################################################
   print('Jouffrey et al. analysis', flush=True)
   ###############################################################################################
   if description_parameters.startswith('polyomino') or 'synthetic_normal_100_15_2' in type_map: #only do this for special cases
      for threshold_phenotype_set in [0.05, 0.1, 0.25]:
         jouffreyGPmapdata_filename =  filepath + 'data/jouffreyGPmapdata_filename'+description_parameters_kbT + 'threshold_phenotype_set' + str(threshold_phenotype_set)+'.csv'
         if not isfile(jouffreyGPmapdata_filename):
            Psetrobustness, Psetevolvability, Psetrobustevolvability = synthetic.ND_GPmapproperties_Jouffrey_def_single_iteration(list_all_possible_structures, K, L, get_Boltzmann_ensemble_givenarray, threshold_phenotype_set)
            df_jouffrey = pd.DataFrame.from_dict({'phenotype': list_all_possible_structures, 'setrob': Psetrobustness, 'setevolv': Psetevolvability, 'setrobustevolv': Psetrobustevolvability})
            df_jouffrey.to_csv(jouffreyGPmapdata_filename)

   ###############################################################################################
   print('find distance between peaks in ND GP map', flush=True)
   ###############################################################################################
   phenos_not_in_GPmap = 0
   if not isfile(filepath + 'data/list_peaks'+description_parameters_kbT+'.csv') or not isfile('data/list_distances_peaks'+description_parameters_kbT+'.npy'):
      phenotype_vs_gmax = {}
      ##
      if not type_map.startswith('RNA'):
         len_list = []
         for p in list_all_possible_structures:
            print(p)
            if type_map.startswith('synthetic') or type_map.startswith('RNA') or type_map.startswith('HP'):
               pmax = np.amax(P_array[..., p])#get_max_P(P_array, p)
               if pmax > 10**-6:
                  #gmax = np.unravel_index(P_array[..., p].argmax(), GPmap.shape)
                  ###
                  gmax_list = np.argwhere(np.abs(P_array[..., p] - pmax ) < 10**-7)
                  len_list.append(len(gmax_list))
                  index = np.random.randint(0, high=len(gmax_list)) 
                  gmax = [gmax_list[index][i] for i in range(L)]
                  assert abs(P_array[tuple([x for x in gmax] + [p,])]- np.amax(P_array[..., p])) < 10**-7
               else:
                  gmax = [-1,] * L
                  print(p, 'not in ND GPmap', description_parameters_kbT)
                  phenos_not_in_GPmap += 1 
            elif type_map.startswith('polyomino'):
               assembly_graph_vs_prob_p =  {a: ensemble[p] for a, ensemble in assembly_graph_no_vs_pheno_ensemble.items() if p in ensemble}
               assembly_graph = max(assembly_graph_vs_prob_p.keys(), key=assembly_graph_vs_prob_p.get)
               geno_assembly_graph = np.where(genotype_to_assembly_graph == assembly_graph)
               randomindex = np.random.randint(0, high=len(geno_assembly_graph[1])) 
               gmax = [geno_assembly_graph[i][randomindex] for i in range(L)]
               assert genotype_to_assembly_graph[tuple(gmax)] == assembly_graph    
            else:
               raise RuntimeError('not implemented', type_map)
            phenotype_vs_gmax[p] = tuple(gmax)
            print('mean number of peaks', np.mean(len(len_list)))
      if type_map.startswith('RNA'):
         phenotype_vs_gmax_array = find_peaks_RNA(P_array)
         phenotype_vs_gmax = {p: tuple(phenotype_vs_gmax_array[p, ...]) for p in list_all_possible_structures if max(phenotype_vs_gmax_array[p, ...]) >= 0}
      
      list_distances_peaks = [synthetic.hamming_dist(phenotype_vs_gmax[p], phenotype_vs_gmax[q]) for i, p in enumerate(list_all_possible_structures) for q in list_all_possible_structures[:i] if max(phenotype_vs_gmax[p]) >= 0 and max(phenotype_vs_gmax[q]) >= 0]
      np.save(filepath + 'data/list_distances_peaks'+description_parameters_kbT+'.npy', list_distances_peaks, allow_pickle=False)
      dict_peaks = {'phenotype': [p for p in list_all_possible_structures], 'peak': ['_'.join([str(i) for i in phenotype_vs_gmax[p]]) for p in list_all_possible_structures]}
      if type_map.startswith('synthetic'):
         dict_peaks['vector'] = ['_'.join([str(i) for i in structure_vs_structure_vect[p]])  for p in list_all_possible_structures]
      pd.DataFrame.from_dict(dict_peaks).to_csv(filepath + 'data/list_peaks'+description_parameters_kbT+'.csv')
      assert phenos_not_in_GPmap <= len(NDph_vs_f) - len([f for f in NDph_vs_f.values() if not np.isnan(f) and f/K**L > 10**(-6)])

   ###############################################################################################
   print('get Pearson correlation between neighbouring/random ensembles', flush=True)
   ###############################################################################################
   filename_corr = filepath + 'data/NDGPmap_genetic_corr_stats'+description_parameters_kbT+'_perpheno.csv'
   if not isfile(filename_corr):
      structure_vs_p_list, structure_vs_p_list_neighbour, structure_vs_p_list_random = {p: [] for p in list_all_possible_structures}, {p: [] for p in list_all_possible_structures}, {p: [] for p in list_all_possible_structures}
      for i in range(10**4):
         g1 = tuple(np.random.choice(K, size=L, replace=True))
         neighbour = synthetic.neighbours_g(g1, K, L)[np.random.choice(L * (K-1))]
         g2 = tuple(np.random.choice(K, size=L, replace=True))
         B1 = get_Boltzmann_ensemble_givenarray(g1)
         Bn = get_Boltzmann_ensemble_givenarray(neighbour)
         B2 = get_Boltzmann_ensemble_givenarray(g2)
         for ph in list_all_possible_structures:
            if ph in B1:
               structure_vs_p_list[ph].append(B1[ph])
            else:
               structure_vs_p_list[ph].append(0)
            if ph in Bn:
               structure_vs_p_list_neighbour[ph].append(Bn[ph])
            else:
               structure_vs_p_list_neighbour[ph].append(0)
            if ph in B2:
               structure_vs_p_list_random[ph].append(B2[ph])
            else:
               structure_vs_p_list_random[ph].append(0)
         
      df_stats = pd.DataFrame.from_dict({'phenotype': list_all_possible_structures, 
                                        'Pearson correlation random': [stats.pearsonr(structure_vs_p_list[s], structure_vs_p_list_random[s])[0] for s in list_all_possible_structures],
                                        'Pearson correlation neighbours': [stats.pearsonr(structure_vs_p_list[s], structure_vs_p_list_neighbour[s])[0] for s in list_all_possible_structures]})
      df_stats.to_csv(filepath + 'data/NDGPmap_genetic_corr_stats'+description_parameters_kbT+'_perpheno.csv')
 
   ###############################################################################################
   print('example plots', flush=True)
   ###############################################################################################
   if not description_parameters.startswith('synthetic'):
      np.random.seed(1)
      #genotype_list_plot = [tuple(np.random.choice(K, size=L, replace=True)) for i in range(15)]
      #for genotype in genotype_list_plot:
      if description_parameters.startswith('RNA'):
         genotype = RNA_functions.sequence_str_to_int('CCUAGCUUGGGU')
         ensemble = RNA_functions.get_Boltzmann_ensemble(genotype, db_to_structure_int=db_to_structure_int, temp=kbT)
         str_list = [ph for ph in sorted(list(ensemble.keys()), key=ensemble.get, reverse=True)]
         df_RNA_example = pd.DataFrame.from_dict({'structure': [structure_int_to_db[s] for s in str_list], 'ensemble prob.': [ensemble[s] for s in str_list]})
         df_RNA_example.to_csv(filepath + 'plots_examples/NDexample_'+description_parameters_kbT+RNA_functions.sequence_int_to_str(genotype)+'.csv')
      elif description_parameters.startswith('ncHP') or description_parameters.startswith('HP'):
         genotype = tuple([{'H': 0, 'P': 1}[x] for x in 'HHHHHPPPHHPPPPHH']) 
         ensemble = HP_functions.HPget_Boltzmann_freq(genotype, contact_maps_as_single_array=contact_maps_as_single_array, number_s_with_cm_list=np.array(number_s_with_cm_list), 
                                                           kbT=kbT, renormalise = True, potential=potential)
         ph_to_plot = [ph for ph in sorted(list(ensemble.keys()), key=ensemble.get, reverse=True) if ensemble[ph] > 0.01]
         if len(ph_to_plot) > 1:
            f, ax = plt.subplots(ncols=len(ph_to_plot), figsize=(3.2* len(ph_to_plot), 2.5))
            for i, ph in enumerate(ph_to_plot):
               HP_functions.plot_structure(structure_int_to_ud[ph], ax[i])
               ax[i].set_title(str(round(ensemble[ph], 3)))
            f.savefig(filepath + 'plots_examples/NDexample_'+description_parameters_kbT+''.join([{0: 'H', 1: 'P'}[x] for x in genotype])+'.pdf')
      elif description_parameters.startswith('polyomino'):
         if L < 12 or param.polyomino_threshold != 5:
            continue
         #assembly_graph_chosen = np.random.choice([a for a, ensemble in assembly_graph_no_vs_pheno_ensemble.items() if 0 != max(list(ensemble.keys()), key=ensemble.get) and 0.4  < max(list(ensemble.values())) < 0.95])
         #genotype = polyomino_functions.find_genotype_from_int(assembly_graph_chosen, genotype_to_assembly_graph)
         genotype = (0, 0, 1, 1, 0, 0, 2, 1, 0, 0, 2, 1)
         ensemble = polyomino_functions.find_full_ensemble_polyomino(genotype, n_runs=10**4, seeded=param.polyomino_seeded_assembly, threshold=param.polyomino_threshold)
         ph_to_plot = [ph for ph in sorted(list(ensemble.keys()), key=ensemble.get, reverse=True) if ensemble[ph] > 0.01 and str(ph) != '0']
         if len(ph_to_plot) > 1:
            f, ax = plt.subplots(ncols=len(ph_to_plot), figsize=(3.2* len(ph_to_plot), 2.5))
            for i, ph in enumerate(ph_to_plot):
               tile_pheno = polyomino_functions.from_str_to_tuple_list(ph)
               polyomino_functions.plot_outline(tile_pheno, ax[i])
               ax[i].set_title(str(round(ensemble[ph], 3)))
            f.savefig(filepath + 'plots_examples/NDexample_'+description_parameters_kbT+''.join([str(x) for x in genotype])+'.pdf')
   ###############################################################################################
   print('plot all polyomino phenotypes', flush=True)
   ###############################################################################################
   if type_map.startswith('polyomino'):
      list_ND_phenos = sorted([ph for ph in list_all_possible_structures if ph not in phenos_Dmap], key=NDph_vs_f.get, reverse=True)
      phenos_Dmap = sorted(phenos_Dmap, key=NDph_vs_f.get, reverse=True)
      for i, list_to_plot in enumerate([list_ND_phenos, phenos_Dmap]):
         f, ax = plt.subplots(ncols=10, nrows = len(list_to_plot)//10 + 1, figsize=(20, len(list_to_plot)/5+2.2))
         for j, ph in enumerate(list_to_plot):
            ph_str = [pdetailed for pdetailed, p in pheno_vs_num.items() if ph == p][0]
            tile_pheno = polyomino_functions.from_str_to_tuple_list(ph_str)
            if len(list_to_plot)//10 > 0:
               indices = (j//10, j%10)
            else:
               indices = j
            polyomino_functions.plot_outline(tile_pheno, ax[indices])
            ax[indices].set_title(str(int(ph))+' '+'rho='+str(round(NDph_vs_rho[ph], 1))+', ev='+str(round(NDPevolv_dict[ph])), fontsize=7)
         for j in range(len(list_to_plot), 10*(len(list_to_plot)//10 + 1)):
            if len(list_to_plot)//10 > 0:
               indices = (j//10, j%10)
            else:
               indices = j
            ax[indices].axis('off')
         f.savefig(filepath + 'plots_examples/'+['NDphenos', 'Dphenos'][i]+description_parameters+'.png', dpi=200)   



