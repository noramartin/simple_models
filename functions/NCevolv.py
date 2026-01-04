import numpy as np 
from numba import jit
from numba.typed import Dict, List
from numba.core import types
from functions.synthetic_model_functions import find_NCs, neighbours_g

def get_ND_GPmap_with_largest_NC_per_pheno(P_array, list_all_phenotypes, get_Boltzmann_ensemble_funct, structure_invalid_test, threshold=0.001):
   L, K = P_array.ndim - 1, P_array.shape[-2]
   p_vs_NCf, p_vs_NCrho, p_vs_NCev = {}, {}, {}
   for phenoNC in range(1, P_array.shape[-1]):
      array_above_threshold = np.greater_equal(P_array[..., phenoNC], threshold * np.ones((K,)*L))
      if not np.any(array_above_threshold):
      	freq, rho, evolv = np.nan, np.nan, np.nan
      else:
      	NCindex_array_p = find_NCs(True, array_above_threshold)
      	NCindex_vs_total_freq = {NCindex: 0 for NCindex in range(1, 1+ np.max(NCindex_array_p))}
      	for g, NCindex in np.ndenumerate(NCindex_array_p):
         	if int(NCindex) > 0:
         	   NCindex_vs_total_freq[int(NCindex)] += P_array[tuple([x for x in g] + [phenoNC])]
         	else:
           	 	assert P_array[tuple([x for x in g] + [phenoNC])] <= threshold
      	NCindex_largest_component = max([NCindex for NCindex in NCindex_vs_total_freq], key=NCindex_vs_total_freq.get)
      	largestNC = np.argwhere(NCindex_array_p == NCindex_largest_component)
      	assert len(largestNC) > 0
      	freq, rho, evolv = ND_GPmapproperties_NC(phenoNC, largestNC, list_all_phenotypes, K, L, get_Boltzmann_ensemble_funct, structure_invalid_test)
      p_vs_NCf[phenoNC] = freq
      p_vs_NCrho[phenoNC] = rho
      p_vs_NCev[phenoNC] = evolv
   return p_vs_NCf, p_vs_NCrho, p_vs_NCev


def get_ND_GPmap_with_largest_NC_per_pheno_polyomino(assembly_graph_no_vs_pheno_ensemble, genotype_to_assembly_graph, list_all_phenotypes, get_Boltzmann_ensemble_funct, structure_invalid_test, threshold=0.001):
   L, K = genotype_to_assembly_graph.ndim, genotype_to_assembly_graph.shape[-1]
   p_vs_NCf, p_vs_NCrho, p_vs_NCev = {}, {}, {}
   for phenoNC in range(1, 1+max(list_all_phenotypes)):
      assembly_graphs_above_threshold = {a: 1 if phenoNC in ensemble and ensemble[phenoNC] > threshold else 0 for a, ensemble in assembly_graph_no_vs_pheno_ensemble.items()}
      array_above_threshold = np.zeros((K,)*L, dtype=np.uint8)
      for g, a in np.ndenumerate(genotype_to_assembly_graph):
      	if assembly_graphs_above_threshold[a]:
      		array_above_threshold[g] = 1
      if not np.any(array_above_threshold):
      	freq, rho, evolv = np.nan, np.nan, np.nan
      else:
      	NCindex_array_p = find_NCs(1, array_above_threshold)
      	NCindex_vs_total_freq = {NCindex: 0 for NCindex in range(1, 1+ np.max(NCindex_array_p))}
      	for g, NCindex in np.ndenumerate(NCindex_array_p):
         	if int(NCindex) > 0:
         	   NCindex_vs_total_freq[int(NCindex)] += assembly_graph_no_vs_pheno_ensemble[genotype_to_assembly_graph[tuple([x for x in g])]][phenoNC]
         	else:
           	 	assert phenoNC not in assembly_graph_no_vs_pheno_ensemble[genotype_to_assembly_graph[tuple([x for x in g])]] or assembly_graph_no_vs_pheno_ensemble[genotype_to_assembly_graph[tuple([x for x in g])]][phenoNC] <= threshold
      	NCindex_largest_component = max([NCindex for NCindex in NCindex_vs_total_freq], key=NCindex_vs_total_freq.get)
      	largestNC = np.argwhere(NCindex_array_p == NCindex_largest_component)
      	freq, rho, evolv = ND_GPmapproperties_NC(phenoNC, largestNC, list_all_phenotypes, K, L, get_Boltzmann_ensemble_funct, structure_invalid_test)
      p_vs_NCf[phenoNC] = freq
      p_vs_NCrho[phenoNC] = rho
      p_vs_NCev[phenoNC] = evolv
   return p_vs_NCf, p_vs_NCrho, p_vs_NCev


def ND_GPmapproperties_NC(phenoNC, NCgenos, list_all_phenotypes, K, L, get_Boltzmann_ensemble_funct, structure_invalid_test):
   freq = 0
   rho_unnorm = 0
   p2_vs_evolv_prod = np.ones(max(list_all_phenotypes) + 1, dtype='float')
   counter = 0
   no_neighbours = (K-1)*L
   for g in NCgenos:
      Boltz = np.array(get_Boltzmann_ensemble_funct(tuple(g)), dtype='float')
      assert abs(sum(Boltz) - 1) < 0.05
      Boltz_neighbours = np.array([get_Boltzmann_ensemble_funct(g2) for g2 in neighbours_g(g, K, L)], dtype='float')
      assert Boltz_neighbours.shape[0] == no_neighbours
      freq, p2_vs_evolv_prod, rho_unnorm = process_genotype_gNC(phenoNC, Boltz, Boltz_neighbours, freq, p2_vs_evolv_prod, rho_unnorm)               
      counter += 1
      if counter % 10**5 == 0:
         print('finished', round(counter/len(NCgenos) * 100, 2),  '%')
   evolv = sum([1 - p2_vs_evolv_prod[p2] for p2 in list_all_phenotypes if phenoNC != p2 and not structure_invalid_test(p2)])
   print('NC finish ND_GPmapproperties', phenoNC)
   return freq, rho_unnorm/(freq*no_neighbours), evolv

@jit(nopython=True)
def process_genotype_gNC(phenoNC, Boltz, Boltz_neighbours, freq, p2_vs_evolv_prod, rho_unnorm):
    freq += Boltz[phenoNC]
    assert Boltz[phenoNC] > 0
    for neighbour in range(Boltz_neighbours.shape[0]):
        Boltz2 = Boltz_neighbours[neighbour, :]
        for p2, B2 in enumerate(Boltz2):
            if p2 > 0.5:
                if p2 != phenoNC:
                     p2_vs_evolv_prod[p2] = p2_vs_evolv_prod[p2]* (1- B2*Boltz[phenoNC])
                elif p2 == phenoNC:
                     rho_unnorm += Boltz[phenoNC] * B2
    return freq, p2_vs_evolv_prod, rho_unnorm
    
############################################################################################################
if __name__ == "__main__":
   from synthetic_model_functions import find_NCs, neighbours_g
   geno_to_ensemble = {(0, 0, 0): {1:0.1, 0: 0.4, 2: 0.5},
                        (1, 0, 0): {1:0.5, 0: 0.3, 2: 0.2},
                        (0, 1, 0): {1:0.4, 0: 0.4, 2: 0.2},
                        (0, 0, 1): {1:0.5, 0: 0.5},
                        (1, 1, 0): {1:0.3, 0: 0.3, 2: 0.4},
                        (1, 0, 1): {1:0.1, 0: 0.3, 2: 0.6},
                        (0, 1, 1): {0: 0.8, 2: 0.2},
                        (1, 1, 1): {1:0.5, 0: 0.3, 2: 0.2}}
   P_array = np.zeros((2, 2, 2, 3))
   for g, ensemble in geno_to_ensemble.items():
      for p, P in ensemble.items():
         P_array[tuple([x for x in g] + [p])] = P
   threshold = 0.22
   for phenoNC in range(1, P_array.shape[-1]):
      array_above_threshold = np.greater_equal(P_array[..., phenoNC], threshold * np.ones((2,)*3))  
      assert np.any(array_above_threshold)
      for g, above_threshold in np.ndenumerate(array_above_threshold):
         assert (above_threshold and geno_to_ensemble[g][phenoNC] > threshold) or ((phenoNC not in geno_to_ensemble[g] or geno_to_ensemble[g][phenoNC] < threshold) and not above_threshold)
      NCindex_array_p = find_NCs(True, array_above_threshold)
      for NCindex in range(1, np.max(NCindex_array_p) + 1):
         print(phenoNC, 'NC', NCindex, [g for g, n in np.ndenumerate(NCindex_array_p) if n == NCindex])



