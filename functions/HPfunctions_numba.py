import numpy as np 
from itertools import product
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from copy import deepcopy
from os.path import isfile, realpath
import pandas as pd
from numba import jit
from collections import Counter
from math import isqrt


direction_to_int = {(1, 0): 0, (-1, 0): 1, (0, 1): 2, (0, -1): 3}
int_to_direction = {i: d for d, i in direction_to_int.items()}
index_to_new_index_90deg = {old_index: direction_to_int[(-old_dir[1], old_dir[0])] for old_dir, old_index in direction_to_int.items()} #rotation by 90
mirror_image_x_axis = {0: 0, 1: 1, 2: 3, 3: 2}
contact_energies = np.array([[-1, 0], [0, 0]], dtype='int')
contact_energies_Li = np.array([[-2.3, -1], [-1, 0]], dtype='float')


def enumerate_all_structures_or_from_file(L, contact_map_list_filename, compact):
	if compact:
		n = isqrt(L)
		assert n**2 == L
	else:
		n = L
	if not isfile(contact_map_list_filename):
		## for L=18, number of self-avoiding walks and contact sets (5808335 and 170670) match the Irbäck paper

		updown_list = [deepcopy(structure_up_down_notation) for structure_up_down_notation in enumerate_all_structures(n,compact=compact) ]
		contact_map_list = [up_down_to_contact_map(s) for s in updown_list]
		print('number of structures', len(contact_map_list))
		contact_map_list_vs_count = Counter(contact_map_list)
		contact_map_vs_updown = {deepcopy(c): deepcopy(s) for c, s in zip(contact_map_list, updown_list)}
		unique_CM_list = [CM for CM in contact_map_list_vs_count]
		print('number of unique contact maps', len(unique_CM_list))
		number_s_with_cm_list = [contact_map_list_vs_count[CM] for CM in unique_CM_list]
		unique_updown_list = [contact_map_vs_updown[CM] for CM in unique_CM_list]
		df_CM = pd.DataFrame.from_dict({'lower contacts of structure': ['_'+'_'.join([str(c[0]) for c in CM]) for CM in unique_CM_list], 
   	                               'corresponding upper contacts of structure': ['_'+'_'.join([str(c[1]) for c in CM]) for CM in unique_CM_list],
   	                               'up-down string of structure': ['_'+'_'.join([str(c) for c in s]) for s in unique_updown_list],
   	                                'number of structures': number_s_with_cm_list})
		if contact_map_list_filename:
			df_CM.to_csv(contact_map_list_filename)
	else:
		df_CM = pd.read_csv(contact_map_list_filename)
		unique_CM_list = [tuple([(int(c0), int(c1)) for c0, c1 in zip(row['lower contacts of structure'].strip('_').split('_'), row['corresponding upper contacts of structure'].strip('_').split('_'))]) if len(row['lower contacts of structure']) > 1 else tuple([]) for rowi, row in df_CM.iterrows()]
		unique_updown_list = [[int(ud) for ud in s.strip('_').split('_')] for s in df_CM['up-down string of structure'].tolist()]
		number_s_with_cm_list = [int(s) for s in df_CM['number of structures'].tolist()]
	if compact:
		n_contacts = list(set([len(cm) for cm in unique_CM_list]))
		assert len(n_contacts) == 1 and n_contacts[0] == (n**2*2 - n*4 + 2)//2
	print('number of contact maps with zero contacts', sum([i for c, i in zip(unique_CM_list, number_s_with_cm_list) if len(c) == 0]))
	print('number_s_with_cm_list_from', min(number_s_with_cm_list), 'to', max(number_s_with_cm_list))
	return unique_CM_list, unique_updown_list, number_s_with_cm_list


def enumerate_all_structures(n, compact=True):
	print('enumerate_all_structures', n)
	if compact:
		L = n**2
		structures = []
		for startposition in product(np.arange(n), repeat=2):
			if sum(startposition)%2 == 0 or n%2 == 0: 
				# only even parity start positions if odd n (source: Sam Greenbury's thesis who relies on Kloczkowski and Jernigan 1997)
				# a simple way of seeing that this has to hold: an odd-length chain will start and end on the same parity, 
				# but a 5x5 lattice has more sites of even parity -> need to start and end at even parity
				structures += find_all_walks_given_start(tuple(startposition), n, L, [startposition,], compact=compact)
	else:
		L = n
		structures = find_all_walks_given_start((0, 0), n, L, [(0, 0),], compact=compact)
	structures_nonredundant = list(set([from_coordinates_to_up_down(s, L) for s in structures]))
	print('take our mirror images', len(structures_nonredundant))
	structures_up_down_notation = take_out_mirror_images(structures_nonredundant)
	print('after removal of mirror images', len(structures_up_down_notation))
	return structures_up_down_notation

def take_out_mirror_images(structure_list):
	#a flip across x-axis is same structure, just turned
	mirror_images = [min(s, tuple([mirror_image_x_axis[i] for i in s])) for s in structure_list] # the mirrored structures still have their first direction aligned with the x-axis because we flip over x
	return list(set(mirror_images))


def turn_such_that_first_index_zero(structure_up_down_notation):
	while structure_up_down_notation[0] != 0:
		structure_up_down_notation = [index_to_new_index_90deg[i] for i in structure_up_down_notation]
	return tuple(structure_up_down_notation)

def from_coordinates_to_up_down(s, L):
	return turn_such_that_first_index_zero([direction_to_int[(s[i + 1][0] - s[i][0], s[i + 1][1] - s[i][1])]  for i in range(L - 1)])


def find_all_walks_given_start(startposition, n, L, walk_so_far, compact=True):
	if len(walk_so_far) == L:
		return [deepcopy(walk_so_far), ]
	list_walks = []
	for next_move in ((1, 0), (-1, 0), (0, 1), (0, -1)):
		new_pos = (startposition[0] + next_move[0], startposition[1] + next_move[1]) 
		if compact and max(new_pos) <= n - 1 and min(new_pos) >= 0 and new_pos not in walk_so_far:
			list_walks += find_all_walks_given_start(new_pos, n, L, deepcopy(walk_so_far + [new_pos,]), compact=compact)
		elif not compact and new_pos not in walk_so_far:
			list_walks += find_all_walks_given_start(new_pos, n, L, deepcopy(walk_so_far + [new_pos,]), compact=compact)
	return list_walks


def plot_structure(structure_up_down_notation, ax):
	ax.axis('off')
	ax.axis('equal')
	current_point = (0, 0)
	ax.scatter([current_point[0], ], [current_point[1],], c='r')
	for d in structure_up_down_notation:
		new_pos = (current_point[0] + int_to_direction[d][0], current_point[1] + int_to_direction[d][1]) 
		ax.plot([current_point[0], new_pos[0]], [current_point[1], new_pos[1]], c='k')
		current_point = (new_pos[0], new_pos[1])

def up_down_to_contact_map(structure_up_down_notation):
	current_point = (0, 0)
	structure_coordinate_notation = [(0, 0)]
	for d in structure_up_down_notation:
		current_point = (current_point[0] + int_to_direction[d][0], current_point[1] + int_to_direction[d][1]) 
		structure_coordinate_notation.append((current_point[0], current_point[1]))
	contact_map = []
	for i, coordi in enumerate(structure_coordinate_notation):
		for j, coordj in enumerate(structure_coordinate_notation):
			if i < j - 1.5 and abs(coordi[0] - coordj[0]) + abs(coordi[1] - coordj[1]) == 1:
			   contact_map.append((i, j))
	return tuple(sorted(contact_map))

def contact_map_to_str(cm):
	return '__'.join([str(i) + '_' + str(j) for i, j in cm])

def free_energy(seq, contact_map, potential='HP'):
	if len(seq) == 25:
		assert len(contact_map) == 16
	if potential == 'HP':
	   return sum([contact_energies[(seq[i], seq[j])] for i, j in contact_map])
	elif potential == 'Li':
		return sum([contact_energies_Li[(seq[i], seq[j])] for i, j in contact_map])	
	else:
		raise RuntimeError('potential', potential, 'not known')

@jit(nopython=True, parallel=True)
def free_energy_list_from_contact_map_array(seq, contact_maps_as_single_array, potential):
	L, nph = len(seq), contact_maps_as_single_array.shape[-1]
	free_energy_array = np.zeros(nph, dtype='float')
	if potential == 'HP':
		for i in range(L):
			for j in range(i):
				for cm_index in range(nph):
					if contact_maps_as_single_array[(i, j, cm_index)] > 0.5:
						free_energy_array[cm_index] = free_energy_array[cm_index] + contact_energies[seq[i], seq[j]]
	elif potential == 'Li':
		for i in range(L):
			for j in range(i):
				for cm_index in range(nph):
					if contact_maps_as_single_array[(i, j, cm_index)] > 0.5:
						free_energy_array[cm_index] = free_energy_array[cm_index] + contact_energies_Li[seq[i], seq[j]]
	else:
		print('potential not known', potential)
		assert 1 == 2
	return free_energy_array




@jit(nopython=True)
def find_mfe(seq, contact_maps_as_single_array, number_s_with_cm_list, kbT, potential='HP'):
	# if unique ground state exists, return index of its contact map +1 - else return 0
	free_energy_list = free_energy_list_from_contact_map_array(seq, contact_maps_as_single_array, potential)
	if not np.isnan(kbT):
	   free_energy_list = free_energy_list - kbT * np.log(number_s_with_cm_list)
	#sorted_free_energy_list = sorted(free_energy_list)
	min_G = np.min(free_energy_list)
	mfe_phenotypes = np.argwhere(free_energy_list < min_G + 0.00000001) + 1
	if max(number_s_with_cm_list) < 1.5:
	   return mfe_phenotypes.reshape(mfe_phenotypes.shape[0])
	elif mfe_phenotypes.shape[0] == 1 and number_s_with_cm_list[mfe_phenotypes[0] - 1] < 1.5:
		return mfe_phenotypes.reshape(mfe_phenotypes.shape[0])
	else: #unfolded if that cm corresponds to several structures
		return np.zeros(1, dtype='int64')



@jit(nopython=True, parallel=True)
def HPget_Z_and_exp(seq, contact_maps_as_single_array, number_s_with_cm_list, kbT, renormalise = True, potential='HP', multiplicity=True):
	free_energy_list = free_energy_list_from_contact_map_array(seq, contact_maps_as_single_array, potential)
	min_G = min(free_energy_list)
	if renormalise:
		exp_list = np.exp(-1.0 * (free_energy_list - min_G)/kbT)
	else:
	    exp_list = np.exp(-1.0 * free_energy_list/kbT)
	if multiplicity:
	   exp_list_with_factors = exp_list * number_s_with_cm_list
	else:
		exp_list_with_factors = exp_list
	Z = np.sum(exp_list_with_factors)
	return exp_list_with_factors, Z


def HPget_Boltzmann_freq(seq, contact_maps_as_single_array, number_s_with_cm_list, kbT,  renormalise = True, potential='HP', multiplicity=True):
	exp_list, Z = HPget_Z_and_exp(seq, contact_maps_as_single_array, number_s_with_cm_list, kbT, renormalise = renormalise, potential=potential, multiplicity=multiplicity)
	B = {i +1: e/Z for i, e in enumerate(exp_list)}
	return B


def contact_maps_list_to_single_array(L, contact_map_list, contact_map_list_single_array_filename=''):
	if not isfile(contact_map_list_single_array_filename):
		single_array = np.zeros((L, L, len(contact_map_list)), dtype='uint8')
		for cm_index, cm in enumerate(contact_map_list):
			for i, j in cm:
				single_array[i, j, cm_index] = 1
				single_array[j, i, cm_index] = 1
		if contact_map_list_single_array_filename:
		   np.save(contact_map_list_single_array_filename, single_array)
	else:
		single_array = np.load(contact_map_list_single_array_filename)
	return single_array


############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
	###
	Kloczkowski_result_directed_compact_walks = {3: 5, 4: 69, 5: 1081, 6: 57337} ##numbers match Kloczkowski and Jernigan 1997
	for n in range(3, 6):
		print('\n\n\n------------------\nfind unique structures, compact n=', n)
		unique_CM_list, unique_updown_list, number_s_with_cm_list = enumerate_all_structures_or_from_file(n**2, contact_map_list_filename='', compact=True)
		assert len(unique_CM_list) == Kloczkowski_result_directed_compact_walks[n]






	

