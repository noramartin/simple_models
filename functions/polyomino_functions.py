from copy import deepcopy
import networkx as nx
from collections import Counter
import numpy as np
from numba import jit
import networkx.algorithms.isomorphism as iso
from functions.polyomino_builder_own import build_polyomino

em = iso.categorical_multiedge_match('edgetype', default='')

#@jit(nopython=True)
def adjacancy_matrix_assembly_graph(g, seeded_assembly):
	assert len(g) % 4 == 0
	#### make graph and record key characteristics for quick polymorphism check
	assembly_graph = nx.MultiDiGraph()
	#assembly_graph_adjacancy = np.zeros((len(g), len(g)), dtype='uint8')
	num_tiles = len(g)//4
	number_edges, tile_vs_info = 0, {i: [0,] * 5 for i in range(num_tiles)}
    # edges to show interface ordering
	for tile in range(num_tiles):
	 	for i in range(3):
	 		assembly_graph.add_edge(i + tile *4, i+1 + tile * 4, edgetype = 'internal')
	 		if tile == 0 and seeded_assembly:
	 			assembly_graph.add_edge(i + tile *4, i+1 + tile * 4, edgetype = 'firsttile')
	 	assembly_graph.add_edge(3 + tile *4, tile * 4, edgetype = 'internal')
	for i, interface in enumerate(g):
		## find binding partners 
		if interface > 0:
			binding_partner = interface + [-1, 1][interface % 2]
			for j, interface2 in enumerate(g[:i]):
				if interface2 == binding_partner:
					assembly_graph.add_edge(i, j, edgetype = 'interaction')
					assembly_graph.add_edge(j, i, edgetype = 'interaction')
					number_edges += 1
					if i//4 == j//4:
						tile_vs_info[i//4][-1] += 1
					else:
					   tile_vs_info[i//4][i%4] += 1
					   tile_vs_info[j//4][j%4] += 1
	tile_vs_info_sorted = {tile: tuple(sorted(info[:4]) + [info[4],]) for tile, info in tile_vs_info.items()}
	return assembly_graph, number_edges, deepcopy(tuple(sorted(list(tile_vs_info.values()))))


@jit(nopython=True)
def rotate_tile(tile_to_rotate, rotation, g_array):
	num_tiles = len(g_array)//4
	rotated_tile = np.concatenate((g_array[tile_to_rotate *4 + rotation: (tile_to_rotate + 1) *4], g_array[tile_to_rotate *4:rotation + tile_to_rotate *4]))
	assert len(rotated_tile) == 4
	return np.array([g_array[i + tile *4] if tile != tile_to_rotate else rotated_tile[i] for tile in range(num_tiles) for i in range(4)], dtype='uint8')

@jit(nopython=True)
def cyclic_permutations(g_array):
	num_tiles = len(g_array)//4
	new_genotype_list = []
	for tile_to_rotate in range(num_tiles):
		for rotation in range(4):
			new_genotype_list.append(rotate_tile(tile_to_rotate, rotation, g_array))
	return new_genotype_list

def construct_simplest_genotype(g, max_c):
	num_tiles = len(g)//4
	if num_tiles == 2:
		minimal_genotypes = [replace_by_nonbounding(g2, max_c) for g2 in cyclic_permutations(np.array(g, dtype='uint8'))]
	else:
		minimal_genotypes = [replace_by_nonbounding(g3, max_c) for g2 in cyclic_permutations(np.array(g, dtype='uint8')) for g3 in cyclic_permutations(np.array(g2, dtype='uint8'))] # so that you also twist against each other
	return min([tuple(g[:]) for g in minimal_genotypes])
 

@jit(nopython=True)
def replace_by_nonbounding(g_array, max_c):
	###
	numbers_in_geno = np.zeros(max_c+1, dtype='uint8')
	for i, c in np.ndenumerate(g_array):
		numbers_in_geno[c] = 1
	###
	old_number_vs_new, counter = np.zeros(max_c+1, dtype=np.int8) -1, 1
	if max_c % 2 == 0:
		unbinding_colors = [0]
		old_number_vs_new[0] = 0
	else:
		unbinding_colors = [0, max_c]
		old_number_vs_new[0] = 0
		old_number_vs_new[max_c] = 0			
	for i, c in np.ndenumerate(g_array):
		if old_number_vs_new[c] < -0.5:
			binding_partner = int(c) + [-1, 1][c % 2]
			if numbers_in_geno[binding_partner]:
			   old_number_vs_new[c] = counter
			   old_number_vs_new[binding_partner] = counter +1
			   counter +=2
			else:
			   old_number_vs_new[c] = 0
	new_g = [old_number_vs_new[x] for x in g_array]
	assert new_g[0] <=1 and new_g[1] <= 3 and new_g[2] <= 5
	return new_g





def find_assembly_graph(g, assembly_graph_list, graph_vs_characteristics, seeded_assembly = False):
	#### make graph and record key characteristics for quick polymorphism check
	assembly_graph, number_edges, tile_vs_info = adjacancy_matrix_assembly_graph(np.array(g), seeded_assembly=seeded_assembly)
	#### polymorphism check: return index to list or append to list
	for i, assembly_graph_existing in enumerate(assembly_graph_list):
		if len(graph_vs_characteristics) == 0 or (graph_vs_characteristics[i][0] == number_edges and graph_vs_characteristics[i][1] == tile_vs_info):
			if nx.is_isomorphic(assembly_graph_existing, assembly_graph, edge_match = em):
				return i + 1, assembly_graph_list, graph_vs_characteristics
	assembly_graph_list.append(deepcopy(assembly_graph))
	graph_vs_characteristics[len(assembly_graph_list) - 1] = (number_edges, tile_vs_info)
	return len(assembly_graph_list), assembly_graph_list, graph_vs_characteristics


@jit(nopython=True)
def find_genotype_from_int(assembly_graph_int, genotype_to_assembly_graph):
	for i, c in np.ndenumerate(genotype_to_assembly_graph): 
		if c == assembly_graph_int:
			return tuple(i[:])



def find_full_ensemble_polyomino(genotype, n_runs, seeded=False, threshold=5):
	graphs_obtained, pos_list = [], []
	index_vs_count = {}
	for i in range(n_runs):
		pos_list_tiles = deepcopy(list(build_polyomino(genotype, threshold=(len(genotype)**2)//2, seeded=seeded))) #default size threshold
		if len(pos_list_tiles) == 0: #size threshold reached:
			try:
		  		index_vs_count[0] += 1
			except KeyError:
				index_vs_count[0] = 1
			continue	
		assert len(pos_list_tiles) > 0		
		### unique representation
		phenotype = min([from_tuple_list_to_str(rotate_coords(pos_list_tiles, angle)) for angle in [0, 90, 180, 270] ])#for reflection in [True, False]
		try:
		   index_vs_count[phenotype] += 1
		except KeyError:
		   index_vs_count[phenotype] = 1
	discarded_count = sum([count for count in index_vs_count.values() if count < threshold])/n_runs
	assert sum([count for count in index_vs_count.values()]) == n_runs
	ensemble = {i: c/n_runs for i, c in index_vs_count.items() if c >= threshold}
	if discarded_count > 0.5/n_runs:
		try:
			ensemble[0] += discarded_count
		except KeyError:
			ensemble[0] = discarded_count
	assert abs(sum(ensemble.values()) - 1) < 0.1/n_runs
	return ensemble


def assembly_graph_int_to_ensemble(assembly_graph_int, genotype_to_assembly_graph, n_runs, seeded_assembly, threshold=5):
	genotype = find_genotype_from_int(assembly_graph_int, genotype_to_assembly_graph)
	ensemble = find_full_ensemble_polyomino(genotype, n_runs, seeded=seeded_assembly, threshold=threshold)
	print('assembly graphs', assembly_graph_int, 'genotype', genotype, 'ensemble', ensemble)
	return ensemble

def align_pos_list_to_zero(pos_list_tiles):
	min_x, min_y = min([t[0] for t in pos_list_tiles]), min([t[1] for t in pos_list_tiles])
	return tuple(sorted([(t[0] - min_x, t[1] - min_y) for t in pos_list_tiles]))


def rotate_coords(pos_list_tiles, angle):
	if angle == 0:
		rotation_matrix = np.array([[1, 0], [0, 1]], dtype='int')
	elif angle == 90:
		rotation_matrix = np.array([[0, -1], [1, 0]], dtype='int')
	elif angle == 180:
		rotation_matrix = np.array([[-1, 0], [0, -1]], dtype='int')
	elif angle == 270:
		rotation_matrix = np.array([[0, 1], [-1, 0]], dtype='int')
	return align_pos_list_to_zero([tuple(np.matmul(rotation_matrix, t)) for t in pos_list_tiles])

def from_tuple_list_to_array(tile_pheno):
	tile_coords = align_pos_list_to_zero(tile_pheno)
	max_x, max_y = max([pos[0] for pos in tile_coords]), max([pos[1] for pos in tile_coords])
	assert min([pos[0] for pos in tile_coords]) == min([pos[1] for pos in tile_coords]) == 0
	array_pheno = np.zeros((max_x + 1, max_y + 1), dtype='uint8')
	for i, (x, y) in enumerate(tile_coords):
		array_pheno[x, y] = 1
	return array_pheno	

def from_tuple_list_to_str(tile_pheno):
	array_pheno = from_tuple_list_to_array(tile_pheno)
	binstr = []
	number_decimal = int(''.join([str(x) for x in array_pheno.flatten()]),2)
	return str(array_pheno.shape[0]) + '_' + str(array_pheno.shape[1]) + '_' + str(number_decimal)
	
def from_str_to_tuple_list(str_pheno):
	shape = (int(str_pheno.split('_')[0]), int(str_pheno.split('_')[1]))
	bin_number = [int(x) for x in bin(int(str_pheno.split('_')[-1]))[2:]]
	if len(bin_number) < shape[0] * shape[1]:
		bin_number = [0, ] * (shape[0] * shape[1] - len(bin_number)) + bin_number[:]
	array_pheno = np.reshape(np.array(bin_number), shape)
	return sorted([tuple(g[:]) for g, i in np.ndenumerate(array_pheno) if i > 0.5])

def plot_outline(tile_pheno, ax):
	for (x, y) in tile_pheno:
		ax.plot([x - 0.5, x+0.5], [y+0.5, y+0.5], c='k')
		ax.plot([x - 0.5, x+0.5], [y-0.5, y-0.5], c='k')
		ax.plot([x - 0.5, x-0.5], [y-0.5, y+0.5], c='k')
		ax.plot([x + 0.5, x+0.5], [y-0.5, y+0.5], c='k')
		ax.fill_between([x - 0.5, x+0.5], y1=[y-0.5, y-0.5], y2=[y+0.5, y+0.5], color='grey')
	ax.set_aspect('equal')
	ax.axis('off')


############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
	#####
	#g = [4, 1, 2, 2, 3, 6, 0, 0]
	g = [0,]*8
	graph = find_assembly_graph(g, [], {})[1][0]
	#print(graph.edges())
	####Victor paper
	graph = find_assembly_graph((1, 2, 0, 0, 0, 1, 2, 0), [], {})[1][0]
	graph2 = find_assembly_graph((1, 2, 0, 0, 1, 2, 0, 0), [], {})[1][0]
	em = iso.categorical_multiedge_match('edgetype', default='')
	assert nx.is_isomorphic(graph, graph2, edge_match = em)
	#### Vctor paper
	graph = find_assembly_graph((1, 1, 1, 1, 2, 0, 0, 0), [], {})[1][0]
	graph2 = find_assembly_graph((2, 0, 0, 0, 1, 1, 1, 1), [], {})[1][0]
	assert nx.is_isomorphic(graph, graph2, edge_match = em)
	#### Leonard thesis
	print('\n\nthesis example')
	graph = find_assembly_graph((1,3,5,0, 2,5,0,4, 0,0,0,6), [], {})[1][0]
	print([(e,f) for e, f, d in graph.edges(data=True) if d['edgetype'] == 'interaction' and e < f])
	graph2 = find_assembly_graph((0,2,6,4, 3,0,0,0, 0,5,1,4), [], {})[1][0]
	assert nx.is_isomorphic(graph, graph2, edge_match = em)
	#### Leonard thesis
	print('\n\nthesis example')
	graph3 = find_assembly_graph((0,1,0,0, 5, 4, 0, 2, 0,6,2,3), [], {})[1][0]
	print([(e,f) for e, f, d in graph3.edges(data=True) if d['edgetype'] == 'interaction' and e < f], '\n')
	graph4 = find_assembly_graph((6,3,1,0, 5,0,0,0, 6,2,0,4), [], {})[1][0]
	print([(e,f) for e, f, d in graph4.edges(data=True) if d['edgetype'] == 'interaction' and e < f], '\n')
	print([(e,f) for e, f, d in graph4.edges(data=True) if d['edgetype'] == 'interaction' and e < f], '\n')
	assert nx.is_isomorphic(graph3, graph4, edge_match = em) 
	assert not nx.is_isomorphic(graph, graph3, edge_match = em)
	assert not nx.is_isomorphic(graph4, graph2, edge_match = em)
	########
	#test conversion to array and back
	tuple_list = sorted([(1, 0), (2, 1), (4, 3), (1, 1), (2, 2), (3, 2), (0, 1)])
	print(tuple(from_str_to_tuple_list(from_tuple_list_to_str(np.array(tuple_list)))))
	assert tuple(tuple_list) == tuple(from_str_to_tuple_list(from_tuple_list_to_str(np.array(tuple_list))))
	

