import numpy as np
import matplotlib.pyplot as pyplot
from copy import deepcopy
import pandas as pd
from numba import jit


def hist_normalised(data, bins, ax, color):
	#print('min(bins), min(data), max(bins), max(data)', min(bins), min(data), max(bins), max(data))
	assert min(bins) <= min(data) and max(bins) >= max(data)
	hist, bins = np.histogram(data, bins=bins, density=False)
	widths = np.diff(bins)
	ax.bar(0.5 * (bins[:-1] + bins[1:]), np.divide(hist, len(data)), widths, color=color, alpha=0.5, edgecolor=color)   
	return max(hist)


def save_assembly_graph_no_vs_pheno_ensemble(filename, assembly_graph_no_vs_pheno_ensemble):
	data_to_save = {'assembly graph': [], 'phenotype': [], 'frequency': []}
	for assembly_graph, ensemble in assembly_graph_no_vs_pheno_ensemble.items():
		for p, freq in ensemble.items():
			data_to_save['assembly graph'].append(assembly_graph)
			data_to_save['phenotype'].append(p)
			data_to_save['frequency'].append(freq)
	pd.DataFrame.from_dict(data_to_save).to_csv(filename)

def load_assembly_graph_no_vs_pheno_ensemble(filename):
	df = pd.read_csv(filename)
	assembly_graph_no_vs_pheno_ensemble = {}
	for i, row in df.iterrows():
		try:
			assembly_graph_no_vs_pheno_ensemble[row['assembly graph']][row['phenotype']] = row['frequency']
		except KeyError:
			assembly_graph_no_vs_pheno_ensemble[row['assembly graph']] = {row['phenotype']: row['frequency']}
	return assembly_graph_no_vs_pheno_ensemble

@jit(nopython=True)
def find_peaks_RNA(Parray):
	L = len(Parray.shape) - 1
	p_vs_max_P = np.zeros(Parray.shape[-1])
	p_vs_genotype = np.zeros((Parray.shape[-1], L)) - 1
	for g, P in np.ndenumerate(Parray):
		p = g[-1]
		if p > 0 and p_vs_max_P[p] < P:
			p_vs_max_P[p] = P
			for i in range(L):
				p_vs_genotype[p, i] = g[i]
	return p_vs_genotype


