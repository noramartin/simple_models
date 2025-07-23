

###############################################################################################
#synthetic
###############################################################################################
synthetic_L, synthetic_K = 15, 2
synthetic_filename = '_'+str(synthetic_L) + '_' + str(synthetic_K) 

###############################################################################################
#RNA
###############################################################################################
RNA_L, RNA_K = 12, 4
RNA_filename = 'RNA' + str(RNA_L) 

###############################################################################################
#HP
###############################################################################################
HP_L, HP_K, HP_potential, HP_iscompact = 16, 2, 'Li', True
if HP_iscompact:
	HP_filename = 'HP' + str(HP_L) + '_' + str(HP_K) + HP_potential
else:
	HP_filename = 'ncHP' + str(HP_L) + '_' + str(HP_K) + HP_potential

###############################################################################################
#polyomino
###############################################################################################
polyomino_ntiles, polyomino_K = 3, 3
polyomino_seeded_assembly = False
polyomino_traditionalGPmap = True ##take only deterministic assemblies for D GP map, not most frequent in ND assemblies
polyomino_n_runs, polyomino_threshold = 500, 5 #5
polyomino_filename = 'polyomino' + str(polyomino_ntiles) + '_' + str(polyomino_K) +'_nruns'+str(polyomino_n_runs)+'_threshold'+str(polyomino_threshold) + '_'
if polyomino_seeded_assembly:
	polyomino_filename += 'seeded'
