import numpy as np 
from fractions import Fraction
from itertools import product
from copy import deepcopy
from os.path import isfile
from numba import jit
from numba.typed import Dict, List
from numba.core import types
import networkx as nx 
from collections import Counter
from multiprocessing import Pool
from functools import partial
#from pyinstrument import Profiler



def isundefined_struct_int(struct_int):
   if  struct_int > 0.1:
      return False
   else:
      return True


def get_energy(struct_vect, seq_tuple, K):
   if K == 2:
      seq_tuple2 = tuple([-1 if s == 0 else 1 for s in seq_tuple])
      assert (len(struct_vect) == len(seq_tuple) or len(struct_vect) == len(seq_tuple) + 1) and max(seq_tuple) < K
   else:
      assert (len(struct_vect) == len(seq_tuple) * K or len(struct_vect) == len(seq_tuple) * K + 1)  and max(seq_tuple) < K
   if K == 2 and len(struct_vect) == len(seq_tuple):
      return -1 * np.dot(struct_vect, seq_tuple2)
   elif K == 2 and len(struct_vect) == len(seq_tuple) + 1:
      return -1 * np.dot(struct_vect[:-1], seq_tuple2) + struct_vect[-1]
   elif K > 2 and len(struct_vect) == len(seq_tuple) * K:
      return -1 * sum([struct_vect[i + c * len(seq_tuple)] for i, c in enumerate(seq_tuple)])
   elif K > 2 and len(struct_vect) == len(seq_tuple) * K + 1:
      return -1 * sum([struct_vect[i + c * len(seq_tuple)] for i, c in enumerate(seq_tuple)]) + struct_vect[-1]
   else:
      raise RuntimeError('need K >= 2')

 
def int_to_vector_structure_random(L, distribution):
   if distribution == 'normal' or distribution == 'normalisednormal' or distribution == 'offsetnormal':
      v = np.random.standard_normal(size=L)
   elif distribution == 'lognormal':
      v = np.random.lognormal(size=L)
   elif distribution == 'uniform':
      v = np.random.uniform(size=L)
   elif distribution == 'binary':
      v = np.random.choice([0.25, 0.75], size=L, replace=True)
   else:
      raise RuntimeError('distribution not implemented')
   return [x for x in v]

def find_structures_that_minimise_G(seq_tuple, structure_vs_structure_vect, resolution, K=2):
   structure_vs_G = {struct: get_energy(struct_vect, seq_tuple, K) for struct, struct_vect in structure_vs_structure_vect.items()}
   min_G = min(structure_vs_G.values())
   return [struct for struct, G in structure_vs_G.items() if abs(G-min_G) < resolution]


def find_energygap(seq_tuple, structure_vs_structure_vect, K=2):
   structure_vs_G = {struct: get_energy(struct_vect, seq_tuple, K) for struct, struct_vect in structure_vs_structure_vect.items()}
   min_G, second_G = sorted([G for G in structure_vs_G.values()])[:2]
   return abs(second_G - min_G)

def get_phenotype_ensemble(seq_tuple, structure_vs_structure_vect, kbT=1, name_function='exponential', K=2, cutoff=0):
   structure_vs_G = {struct: get_energy(struct_vect, seq_tuple, K) for struct, struct_vect in structure_vs_structure_vect.items()}
   if name_function in ['ReLu', 'Softplus']:
      structure_vs_Gscaled = {structure:  -1*G + kbT  for structure, G in structure_vs_G.items()} 
   if name_function in ['exponential', 'inversesquared', 'expsquared']:
      Gmin = np.min([G for G in structure_vs_G.values()])
   if name_function  == 'exponential':
      dict_unnormalised =  {structure: np.exp(-1 * (G-Gmin)/kbT) for structure, G in structure_vs_G.items()} #Gmin to avoid overflow errors
   elif name_function == 'linear':
      Gmax = max(structure_vs_G.values())
      dict_unnormalised = {structure:   (Gmax - G + kbT)  for structure, G in structure_vs_G.items()} #needs to be positive
   elif name_function == 'inversesquared':
      dict_unnormalised = {structure: 1/(G - Gmin + kbT)**2  for structure, G in structure_vs_G.items()}
   elif name_function == 'expsquared':
      dict_unnormalised = {structure: np.exp(-1 * (G - Gmin)**2/kbT)  for structure, G in structure_vs_G.items()}   
   elif name_function == 'ReLu':
      dict_unnormalised = {structure: geff if geff > 0 else 0 for structure, geff in structure_vs_Gscaled.items()}
   elif name_function == 'Softplus':   
      dict_unnormalised = {structure: np.log(1+np.exp(geff)) if geff < 10 else geff for structure, geff in structure_vs_Gscaled.items()}
   Z = sum([i for i in dict_unnormalised.values()])
   assert min([k for k in dict_unnormalised.keys()]) > 0 
   assert min([k for k in dict_unnormalised.values()]) >= 0 
   if Z < 10**(-9):
      assert name_function == 'ReLu'
      return {0: 1}
   assert len([i for i in dict_unnormalised.values() if np.isnan(i)]) == 0
   assert Z > 0
   Boltz_dist = {structure: G/Z for structure, G in dict_unnormalised.items() if G/Z > cutoff}
   if abs(sum(Boltz_dist.values()) - 1) > 0.0001:
      assert sum(Boltz_dist.values()) < 1
      Boltz_dist[0] = 1 - sum(Boltz_dist.values())
   abs(sum(Boltz_dist.values()) - 1) < 0.0001
   return {B:P for B, P in Boltz_dist.items() if P > cutoff}

def get_frequencies_in_array(GPmap, ignore_undefined=True, structure_invalid_test=isundefined_struct_int):
   """ sum over an entire GP array: and sum the number of times each structure is found;
   if ignore_undefined=True, the function structure_invalid_test will determine, which structures are selected; 
   otherwise all structures will be selected"""
   ph_vs_f = {}
   for p in GPmap.copy().flat:
      try:
         ph_vs_f[p] += 1
      except KeyError:
         ph_vs_f[p] = 1
   if not ignore_undefined:
      return ph_vs_f
   else:
      return {p:f for p, f in ph_vs_f.items() if not structure_invalid_test(p)}

def get_deterministic_Prho_Pevolv(GPmap, structure_invalid_test):
   print( 'mutational neighbourhood - rh o and evolv')
   L, K = GPmap.ndim, GPmap.shape[0]
   rho_phenotype_unnorm = {}
   Pneighbours = {}
   freq = {}
   for genotype, ph in np.ndenumerate(GPmap):   
      if not structure_invalid_test(ph):
         neighbours = neighbours_g(tuple(genotype), K, L)
         assert len(neighbours) == L * (K-1)
         try:
            freq[ph] += 1
         except KeyError:
            freq[ph] = 1
         for neighbourgeno in neighbours:
            neighbourpheno = GPmap[tuple(neighbourgeno)]
            if neighbourpheno == ph:
               try:
                  rho_phenotype_unnorm[ph] += 1
               except KeyError:
                  rho_phenotype_unnorm[ph] = 1
            elif not structure_invalid_test(neighbourpheno):
               try:
                  Pneighbours[ph].add(neighbourpheno)
               except KeyError:
                  Pneighbours[ph] = set([neighbourpheno,])
   rho = {p: rho_phenotype_unnorm[p]/(f * L * (K-1)) if p in rho_phenotype_unnorm else 0 for p, f in freq.items()}
   evolv = {p: len(Pneighbours[p]) if p in Pneighbours else 0 for p in freq}
   return freq, rho, evolv


def get_Grobustness_Gevolvability(GPmap, structure_invalid_test=isundefined_struct_int):
   L, K = GPmap.ndim, GPmap.shape[0]
   Grobustness, Gevolvability = np.zeros_like(GPmap, dtype=float), np.zeros_like(GPmap, dtype='float') #uint8')
   for genotype, ph in np.ndenumerate(GPmap):
      if isundefined_struct_int(ph):
         Grobustness[genotype] = np.nan
         Gevolvability[genotype] = np.nan         
      else:
         neighbours = neighbours_g(tuple(genotype), K, L)
         neighbourphenos = [GPmap[tuple(deepcopy(neighbourgeno))] for neighbourgeno in neighbours]
         assert len(neighbourphenos) == L * (K-1)
         Grobustness[genotype] = neighbourphenos.count(ph)/float(len(neighbourphenos))
         Gevolvability[genotype] = len(set([n for n in set(neighbourphenos) if n != ph and not isundefined_struct_int(n)]))
   return Grobustness, Gevolvability


def neighbours_g(g, K, L): 
   """list all pont mutational neighbours of sequence g (integer notation)"""
   return [tuple([oldK if gpos!=pos else new_K for gpos, oldK in enumerate(g)]) for pos in range(L) for new_K in range(K) if g[pos]!=new_K]

def neighbours_g_site(g, site, K): 
   """list all pont mutational neighbours of sequence g (integer notation)"""
   return [tuple([oldK if gpos!=site else new_K for gpos, oldK in enumerate(g)]) for new_K in range(K) if g[site]!=new_K]

###############################################################################################
  
def get_sequence_constraints(chosen_ph, GPmap):
   L, K = GPmap.ndim, GPmap.shape[0]
   neutral_set = np.argwhere(GPmap == chosen_ph)
   geno_vs_NC_array = find_NC(deepcopy(tuple(neutral_set[np.random.randint(len(neutral_set))])), GPmap)
   g0_list = [tuple(g0) for g0, NC in np.ndenumerate(geno_vs_NC_array) if NC > 0.5]
   if len(g0_list) == 0:
      return [np.nan for i in range(L)], [np.nan for i in range(L)], len(g0_list), grho_list, {}
   test_NC(g0_list, GPmap)
   pos_vs_versatility_list = {i: [] for i in range(L)}
   grho_list = []
   print('number of genotypes in NC', len(g0_list))
   for genotype in g0_list:
      assert GPmap[genotype] == chosen_ph
      grho = 0
      for i in range(L):
         neighbourphenos = [GPmap[tuple(deepcopy(neighbourgeno))] for neighbourgeno in neighbours_g_site(tuple(genotype), i, K)]
         pos_vs_versatility_list[i].append(neighbourphenos.count(chosen_ph))
         grho += neighbourphenos.count(chosen_ph)
      grho_list.append(grho)
   no_neutral_vs_prevalence = {i: [0 for i in range(L)] for i in range(K)}
   for pos in range(L):
      for i in range(K):
         no_neutral_vs_prevalence[i][pos] = pos_vs_versatility_list[pos].count(i)
      print('seq constraints', pos, Counter(pos_vs_versatility_list[pos]))
      assert sum([no_neutral_vs_prevalence[i][pos] for i in range(K)]) == len(g0_list)
   return [np.mean(pos_vs_versatility_list[i]) for i in range(L)], [np.std(pos_vs_versatility_list[i]) for i in range(L)], len(g0_list), grho_list, no_neutral_vs_prevalence

def test_NC(g0_list, GPmap):
   for g in g0_list:
      assert GPmap[g] == GPmap[g0_list[0]]
   G = nx.Graph()
   for i, g in enumerate(g0_list):
      for j, h in enumerate(g0_list):
         if i < j:
            if hamming_dist(g, h) == 1:
               G.add_edge(tuple(deepcopy(g)), tuple(deepcopy(h)))
   assert nx.is_connected(G)
   print('NC test complete')

def find_NC(g0, GPmap):
   #using the algorithm from Gruener et al. (1996); adapted from biomorphs paper
   K, L = GPmap.shape[1], GPmap.ndim
   structure_int = GPmap[g0]
   print('find NC of structure', structure_int, flush=True)
   U_list, U, NCindex_array = [], np.zeros_like(GPmap, dtype='uint16'), np.zeros_like(GPmap, dtype='uint16')
   U[tuple(g0)] = 1
   U_list.append(tuple(g0))
   while len(U_list)>0: #while there are elements in the unvisited list
      g1 = deepcopy(U_list[0] )
      assert GPmap[g1] == structure_int
      for g2 in neighbours_g(g1, K, L):
         ph2 = int(GPmap[tuple(g2)])
         if ph2 == structure_int and U[tuple(g2)] < 0.5 and NCindex_array[tuple(g2)] < 0.5:
            U[tuple(g2)] = 1
            U_list.append(tuple(g2[:]))
      U[tuple(g1)] = 0
      NCindex_array[tuple(g1)] = 1 #visited list
      U_list.remove(tuple(g1))
   return NCindex_array


def find_NCs(structure_int, GPmap):
   #using the algorithm from Gruener et al. (1996); adapted from biomorphs paper
   print('find NC of structure', structure_int, flush=True)
   K, L = GPmap.shape[1], GPmap.ndim
   all_NCs_found = False 
   NCindex_array = np.zeros_like(GPmap, dtype='uint16') #zero means not in set - initialise U as both array and list for faster lookup
   NCindex = 1
   for g, ph in np.ndenumerate(GPmap):
      if ph == structure_int and NCindex_array[g] < 0.5:
         g0 = tuple(g[:])
         continue
   while not all_NCs_found:
      U_list, U = [], np.zeros_like(GPmap, dtype='uint16')
      U[tuple(g0)] = 1
      U_list.append(tuple(g0))
      while len(U_list)>0: #while there are elements in the unvisited list
         g1 = U_list.pop() #deepcopy(U_list[0] )
         assert GPmap[g1] == structure_int
         for g2 in neighbours_g(g1, K, L):
            ph2 = int(GPmap[tuple(g2)])
            if ph2 == structure_int and U[tuple(g2)] < 0.5 and abs(int(NCindex_array[tuple(g2)]) - NCindex) > 0.5:
               U[tuple(g2)] = 1
               U_list.append(tuple(g2[:]))
               assert NCindex_array[tuple(g2)] < 0.5
         U[tuple(g1)] = 0
         NCindex_array[tuple(g1)] = NCindex #visited list
         #U_list.remove(tuple(g1))
      ### clean up
      NCindex += 1
      all_NCs_found_new = True 
      for g, ph in np.ndenumerate(GPmap):
         if ph == structure_int and NCindex_array[g] < 0.5:
            all_NCs_found_new = False 
            g0 = tuple(g[:]) 
            continue
      all_NCs_found = all_NCs_found_new
   assert np.array_equal(NCindex_array > 0.5, GPmap == structure_int)
   return NCindex_array


def shape_space_covering(repetitions, GPmap, structure_invalid_test):
   L, K = GPmap.ndim, GPmap.shape[0]
   number_phenos = len([p for p in np.unique(GPmap) if not structure_invalid_test(p)])
   steps_vs_number_found = {i: [] for i in range(1, L)}
   for repetition in range(repetitions):
      startgeno = tuple(list(np.random.choice(np.arange(K), L)))
      for dist in range(1, L):
         phenos_in_range = np.unique([GPmap[g] for g, ph in np.ndenumerate(GPmap) if hamming_dist(g, startgeno) <= dist and not structure_invalid_test(ph)])
         steps_vs_number_found[dist].append(len(phenos_in_range)/number_phenos)
   return [np.mean(steps_vs_number_found[i]) for i in range(1, L)], [np.std(steps_vs_number_found[i]) for i in range(1, L)] 

###############################################################################################

def hamming_dist(g1, g2):
   assert len(g1) == len(g2)
   return len([x for i, x in enumerate(g1) if x != g2[i]])


def get_prob_lowest_G(K, L, get_Boltzmann_ensemble, type_prob='mfe'):
   assert type_prob == 'mfe' or type_prob == 'first_suboptimal'
   mfe_p_array = np.zeros((K,)*L, dtype=float)
   count = 0
   for g, p in np.ndenumerate(mfe_p_array):
      Boltz = get_Boltzmann_ensemble(g)
      assert len([x for x in Boltz.values() if np.isnan(x)]) == 0 #should only have zeros and floats
      if type_prob == 'mfe':
         mfe_p_array[g] = np.nanmax(list(Boltz.values()))
         if mfe_p_array[g] == 0:
            print('zero max')
      elif type_prob == 'first_suboptimal':
         if len(Boltz.keys()) >= 2:
            mfe_p_array[g] = sorted(list(Boltz.values()))[-2]
         else:
            mfe_p_array[g] = np.nan
      count += 1
      if count% 10**4 == 0:
         print('finished', 'get_prob_lowest_G', type_prob, count/K**L * 100, '%')
   return mfe_p_array



#@jit(nopython=True)
def get_entropy(p_list):
   return np.sum([-1 * p * np.log(p) for p in p_list if p > 0])

def get_Boltzmann_ensemble(g, P_array):
   B = {s: P_array[tuple([x for x in g] + [s,])] for s in range(1, P_array.shape[-1])}
   return B


@jit(nopython=True)
def get_Boltzmann_ensemble_list(g, P_array):
   B = P_array[g].flatten()
   #B1 = [P_array[tuple([x for x in g] + [s,])] for s in range(P_array.shape[-1])]
   #assert abs(sum(B-B1)) < 10**-3
   return B


def get_Boltzmann_ensemble_list_polyomino(g, g_vs_assembly_graph, assembly_graph_vs_ensemble_list):
   return assembly_graph_vs_ensemble_list[g_vs_assembly_graph[g]]


def ND_GPmapproperties_Jouffrey_def_single_iteration(list_all_phenotypes, K, L, get_Boltzmann_ensemble, threshold_set):
   ####### input can be all folded or all phenotypes
   counter = 0
   ## initial variables pheno freq
   Psetrobustness, Psetevolvability, Psetrobustevolvability, Pnorm  = {p: 0 for p in list_all_phenotypes}, {p: 0 for p in list_all_phenotypes}, {p: 0 for p in list_all_phenotypes}, {p: 0 for p in list_all_phenotypes}
   ##
   for g, dummy in np.ndenumerate(np.zeros((K,)*L, dtype=np.uint8)):
      set_g = set([x for x, P in get_Boltzmann_ensemble(g).items() if P >= threshold_set and x > 0.5])
      Gsetrobustness, Gsetrobustevolvability, Gsetevolvability = 0, 0, 0
      for g2 in neighbours_g(g, K, L):
         Boltz2 = get_Boltzmann_ensemble(g2)
         set_n = set([x for x, P in Boltz2.items() if P >= threshold_set and x > 0.5])
         set_joint = set_n | set_g
         assert len(set_joint) >= max(len(set_g), len(set_n))
         if len(set_g.intersection(set_n)) > 0:
            Gsetrobustness += 1.0/((K-1) * L)
            if len(set_joint) > len(set_g):
                Gsetrobustevolvability += 1.0/((K-1) * L)
         if len(set_joint) > len(set_g):
            Gsetevolvability += 1.0/((K-1) * L)
         del Boltz2
      ####
      for ph in set_g:
         Psetrobustness[ph] += Gsetrobustness
         Psetrobustevolvability[ph] += Gsetrobustevolvability
         Psetevolvability[ph] += Gsetevolvability
         Pnorm[ph] += 1
      ## finished 
      counter += 1
      if counter % 10**5 == 0:
         print('finished', counter/K**L*100, '%', flush=True)
   # final processing - pheno rob
   Psetrobustness_norm = [Psetrobustness[ph]/Pnorm[ph] if Pnorm[ph] > 0 else np.nan for ph in list_all_phenotypes]
   Psetevolvability_norm = [Psetevolvability[ph]/Pnorm[ph] if Pnorm[ph] > 0 else np.nan for ph in list_all_phenotypes]
   Psetrobustevolvability_norm = [Psetrobustevolvability[ph]/Pnorm[ph] if Pnorm[ph] > 0 else np.nan for ph in list_all_phenotypes]
   return Psetrobustness_norm, Psetevolvability_norm, Psetrobustevolvability_norm


def ND_GPmapproperties(list_all_phenotypes, K, L, get_Boltzmann_ensemble_funct, structure_invalid_test):
   print('start ND_GPmapproperties')
   list_all_phenotypes_valid = [p for p in list_all_phenotypes if not structure_invalid_test(p)]
   ph_vs_f = np.zeros(max(list_all_phenotypes) + 1, dtype='float')
   ph_vs_entropy_unnorm = np.zeros(max(list_all_phenotypes) + 1, dtype='float')
   ph_vs_rho_unnorm = np.zeros(max(list_all_phenotypes) + 1, dtype='float')
   ph_vs_p2_vs_evolv_prod = np.ones((max(list_all_phenotypes) + 1, max(list_all_phenotypes) + 1), dtype='float')
   Grobustness, Gevolvability = np.zeros((K,)*L, dtype=float), np.zeros((K,)*L, dtype=float)
   counter = 0
   no_neighbours = (K-1)*L
   for g, dummy in np.ndenumerate(np.zeros((K,)*L, dtype=np.uint8)):
      Boltz = np.array(get_Boltzmann_ensemble_funct(g), dtype='float')
      assert abs(sum(Boltz) - 1) < 0.05
      Boltz_neighbours = np.array([get_Boltzmann_ensemble_funct(g2) for g2 in neighbours_g(g, K, L)], dtype='float')
      assert Boltz_neighbours.shape[0] == no_neighbours
      grho, gev, ph_vs_entropy_unnorm, ph_vs_f, ph_vs_p2_vs_evolv_prod, ph_vs_rho_unnorm = process_genotype_g(Boltz, Boltz_neighbours, ph_vs_entropy_unnorm, ph_vs_f, ph_vs_p2_vs_evolv_prod, ph_vs_rho_unnorm, list_all_phenotypes_valid, no_neighbours)               
      assert not np.isnan(grho) and not np.isnan(gev)     
      Grobustness[g] = grho
      Gevolvability[g] = gev
      counter += 1
      if counter % 10**3 == 0:
         print('finished', round(counter/K**L * 100, 2),  '%')
   for p, f in enumerate(ph_vs_f):
      if f < 10**(-3) and p in list_all_phenotypes_valid:
         print('zero-prob phenotype', p)
   ph_vs_rho = {p: rho/(ph_vs_f[p] * no_neighbours) if ph_vs_f[p] > 0 else np.nan for p, rho in enumerate(ph_vs_rho_unnorm) if p in list_all_phenotypes_valid}
   ph_vs_entropy = {p: e/ph_vs_f[p] + np.log(ph_vs_f[p]) if ph_vs_f[p] > 0 else np.nan for p, e in enumerate(ph_vs_entropy_unnorm) if p in list_all_phenotypes_valid}
   ph_vs_evolv = {p: sum([1 - ph_vs_p2_vs_evolv_prod[p][p2] for p2 in list_all_phenotypes if p!= p2 and not structure_invalid_test(p2)]) if ph_vs_f[p] > 0 else np.nan for p in list_all_phenotypes if not structure_invalid_test(p)}
   ph_vs_f = {p: f for p, f in enumerate(ph_vs_f) if p in list_all_phenotypes_valid}
   print('finish ND_GPmapproperties')
   return ph_vs_f, ph_vs_rho, ph_vs_evolv, ph_vs_entropy, Grobustness, Gevolvability

@jit(nopython=True)
def process_genotype_g(Boltz, Boltz_neighbours, ph_vs_entropy_unnorm, ph_vs_f, ph_vs_p2_vs_evolv_prod, ph_vs_rho_unnorm, list_all_phenotypes_valid, no_neighbours):
   grho, gevolv_prod = 0, np.ones((max(list_all_phenotypes_valid) + 1, max(list_all_phenotypes_valid) + 1), dtype='float')
   for p, B in enumerate(Boltz):
      if B > 0 and p > 0.5:
         ph_vs_entropy_unnorm[p] -= B* np.log(B)
         ph_vs_f[p] += B
         for neighbour in range(Boltz_neighbours.shape[0]):
            Boltz2 = Boltz_neighbours[neighbour, :]
            for p2, B2 in enumerate(Boltz2):
               if p2 > 0.5:
                  if p2 != p:
                     gevolv_prod[(p, p2)] *= (1 - B2)
                     ph_vs_p2_vs_evolv_prod[p, p2] = ph_vs_p2_vs_evolv_prod[p, p2]* (1- B2*B)
                  elif p2 == p:
                     ph_vs_rho_unnorm[p] += B * B2
                     grho += B * B2  
   grho = grho/(no_neighbours)
   gev = sum([B * sum([1 - gevolv_prod[(p, p2)] for p2 in list_all_phenotypes_valid if p2 != p]) for p, B in enumerate(Boltz) if p > 0.5])
   return grho, gev, ph_vs_entropy_unnorm, ph_vs_f, ph_vs_p2_vs_evolv_prod, ph_vs_rho_unnorm







############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
   GPmap_schematic_Paula_paper = {(0, 0): {1: 0.5, 2: 0.25, 3: 0.25}, (0, 1): {1: 0.75, 2: 0.25}, (1, 0): {2: 0.75, 3: 0.25}, (1, 1): {3: 0.75, 1: 0.25}}
   K, L = 2, 2
   def get_Boltzmann_ensemble(g):
      return [GPmap_schematic_Paula_paper[tuple(g)][p] if p in GPmap_schematic_Paula_paper[tuple(g)] else 0 for p in range(4)]
   NDph_vs_f_final, prob_final, pevolv_final, ph_vs_entropy, Grobustness, Gevolvability = ND_GPmapproperties([1, 2, 3], K, L, get_Boltzmann_ensemble, structure_invalid_test=isundefined_struct_int)
   print('NDph_vs_f', {p: f/K**L for p, f in NDph_vs_f_final.items()}, '\n---------------\n\n')
   print('p_rob_final', prob_final, '\n---------------\n\n')
   print('pevolv_final', pevolv_final, '\n---------------\n\n')
   print('Grobustness', [(g, x) for g, x in np.ndenumerate(Grobustness)], '\n---------------\n\n')
   print('Gevolvability', [(g, x) for g, x in np.ndenumerate(Gevolvability)], '\n---------------\n\n')
   ########
   GPmap_schematic_slides = {'AA': 'r', 'AT': 'r', 'AC': 'r', 'TA': 'r', 'TT': 'r', 'TC': 'r', 'GG': 'r',
                             'CA': 'b', 'CT': 'b', 'CC': 'b', 'TG': 'b', 'AG': 'b', 'GA': 'g', 'GT': 'g', 'CG': 'y'}
   letter_to_number = {l: i for i, l in enumerate('ACGT')}
   p_to_number = {'r': 1, 'b': 2, 'g': 3, 'y': 4}
   GPmap = np.zeros((4, 4), dtype='int64')
   for g, p in GPmap_schematic_slides.items():
      GPmap[tuple([letter_to_number[l] for l in g])] = p_to_number[p]
   def structure_invalid_test(s):
      return False
   print('robustness', get_deterministic_Prho_Pevolv(GPmap, structure_invalid_test=structure_invalid_test)[0])
   print('evolvability', get_deterministic_Prho_Pevolv(GPmap, structure_invalid_test=structure_invalid_test)[1])
   #####
   from fractions import Fraction
   print('\n\n schematic in paper')
   K, L = 3, 2
   GPmap_schematic_paper = {'AC': {'b': Fraction(3, 4), 'g': Fraction(1, 4)}, 
                           'AA': {'b': Fraction(1,2), 'r': Fraction(1, 4), 'o': Fraction(1, 4)},
                           'AG': {'o': Fraction(3,4), 'b': Fraction(1, 4)},
                           'CC': {'b': Fraction(3, 4), 'g': Fraction(1, 4)},
                           'CA': {'b': Fraction(3, 4), 'r': Fraction(1, 4)},
                           'CG': {'b': Fraction(1, 2), 'r': Fraction(1, 4), 'o': Fraction(1, 4)},
                           'GC': {'g': Fraction(3, 4), 'b': Fraction(1, 4)},
                           'GA': {'r': Fraction(3, 4), 'g': Fraction(1, 4)},
                           'GG': {'r': Fraction(2, 4), 'b': Fraction(1, 4), 'o': Fraction(1, 4)}}
   number_to_letter_seq = {i: c for i, c in enumerate(['A', 'G', 'C'])}
   pheno_str_to_int = {c: i + 1 for i, c in enumerate(['b', 'g', 'r', 'o'])} 
   pheno_int_to_str = {i: c for c, i in pheno_str_to_int.items()}
   def get_Boltzmann_ensemble(g):
      B = GPmap_schematic_paper[''.join([number_to_letter_seq[x] for x in g])]
      return [0,] + [B[pheno_int_to_str[p]] if pheno_int_to_str[p] in B else 0 for p in range(1, 5)]
   NDph_vs_f_final, prob_final, pevolv_final, ph_vs_entropy, Grobustness, Gevolvability = ND_GPmapproperties([1, 2, 3, 4], K, L, get_Boltzmann_ensemble, structure_invalid_test=isundefined_struct_int)
   print('phenotype freq of b', NDph_vs_f_final[1]/K**L, '\n---------------\n\n')
   print('p rob of b', prob_final[1], '\n---------------\n\n')
   print('pevolv_final of b', pevolv_final[1], '\n---------------\n\n')
   print('Grobustness of AC', Grobustness[0, 2], '\n---------------\n\n')
   print('Gevolvability of AC', Gevolvability[0, 2], '\n---------------\n\n')   
   ########
   struct_vect = (0, 0.5, 0.3, 0.4,      0.1, 0.1, 0.2, 0.8,    1.1, 2, 0.3, -0.1)
   G = get_energy(struct_vect, (0, 2, 1, 2), K=3)
   assert abs(G + (0 + 2 + 0.2 - 0.1)) < 0.001
   G = get_energy(struct_vect, (0, 0, 1, 2), K=3)
   assert abs(G + (0 + 0.5 + 0.2 - 0.1)) < 0.001
   G = get_energy(struct_vect, (2, 2, 2, 1), K=3)
   assert abs(G + (1.1+2+0.3+0.8)) < 0.001
   G = get_energy(struct_vect, (1, 1, 1, 1), K=3)
   assert abs(G + (1.2)) < 0.001
   G = get_energy(struct_vect, (1, 1, 2, 1), K=3)
   assert abs(G + (1.3)) < 0.001
   G = get_energy(struct_vect, (0, 2, 1, 2), K=3)
   assert abs(G + (0 + 2+0.2-0.1)) < 0.001
   G = get_energy(struct_vect, (2, 2, 2, 2), K=3)
   assert abs(G + (3.3)) < 0.001

