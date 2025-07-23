import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy

direction_to_vect = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}


def interfaces_bind(i, j):
	if i == 0 or j == 0:
		return False
	elif int(i) + [-1, 1][i % 2] == j:
		return True
	else:
		return False

def return_binding_interfaces(tiles, binding_interface_type):
	return [(i, j) for i, t in enumerate(tiles) for j, char in enumerate(t) if interfaces_bind(char, binding_interface_type)]

def build_polyomino(genotype, seeded, threshold, plot=False):
	assert len(genotype) % 4 == 0
	tiles = [tuple(genotype[4*i: 4*i + 4]) for i in range(len(genotype)//4)]
	#print('tiles', tiles)
	tile_vs_binding_interfaces = {t: [i for i, c in enumerate(tile) if len([c2 for i2, c2 in enumerate(genotype) if interfaces_bind(c2, c)]) > 0] for t, tile in enumerate(tiles)}
	#print('tile_vs_binding_interfaces', tile_vs_binding_interfaces)
	if seeded:
		assembly = [(0, 0, 0, 0)]
	else:
		assembly = [(0, 0, np.random.choice(len(tiles)), 0)]
	new_tile = assembly[0][2]
	free_interfaces = [tile_to_free_interface_info((0,0), 0, i, tiles[new_tile][i]) for i in tile_vs_binding_interfaces[new_tile]]
	occupied = [(0, 0),]
	while len(assembly) < threshold:
		if len(free_interfaces) == 0:		
			return [tuple(a[:2]) for a in assembly]
		######
		new_bindings = get_binding_interfaces_surface(assembly, tiles, genotype, occupied, tile_vs_binding_interfaces)
		chosen_binding = deepcopy(new_bindings[np.random.choice(len(new_bindings))])
		new_pos, new_tile, new_tile_orientation = (chosen_binding[0], chosen_binding[1]), chosen_binding[2], chosen_binding[3]
		###
		#positions = list(set([tuple(n[:2]) for n in new_bindings]))
		#number = [len([n for n in new_bindings if tuple(n[:2]) == pos]) for pos in positions]
		#print('possible positions and number options', positions, number)
		#
		assembly.append((new_pos[0], new_pos[1], new_tile, new_tile_orientation))
		free_interfaces += [tile_to_free_interface_info(new_pos, new_tile_orientation, i, tiles[new_tile][i]) for i in tile_vs_binding_interfaces[new_tile]]
		occupied.append(new_pos)
		free_interfaces = [a for a in free_interfaces if tuple(a[:2]) not in occupied] 
	return []


def get_binding_interfaces_surface(assembly, tiles, genotype, occupied_pos, tile_vs_binding_interfaces):
	#occupied_pos = [tuple(assembly_tile_info[:2]) for assembly_tile_info in assembly]
	new_bindings = []
	#tile_vs_binding_interfaces = {t: [i for i, c in enumerate(tile) if len([c2 for i2, c2 in enumerate(genotype) if interfaces_bind(c2, c)]) > 0] for t, tile in enumerate(tiles)}
	for assembly_tile_info in assembly:
		pos_current_tile = tuple(assembly_tile_info[:2])
		tile_identity_current, rotation_current = assembly_tile_info[2:]
		for binding_interface in tile_vs_binding_interfaces[tile_identity_current]:
			pos_new_x, pos_new_y, contact_point, interface_type = tile_to_free_interface_info(pos_current_tile, rotation_current, binding_interface, tiles[tile_identity_current][binding_interface])
			if (pos_new_x, pos_new_y) not in occupied_pos:
				for b in return_binding_interfaces(tiles, interface_type):
					new_tile_orientation = (4 + contact_point - b[-1])%4
					new_bindings.append(tuple([pos_new_x, pos_new_y, b[0], new_tile_orientation]))
	return list(set(new_bindings))




def tile_to_free_interface_info(pos, orientation, interface, interface_type):
	direction_new_tile = (orientation + interface)%4
	pos_new = (pos[0] + direction_to_vect[direction_new_tile][0], pos[1] + direction_to_vect[direction_new_tile][1])
	contact_point = {0: 2, 1:3, 3:1, 2:0}[direction_new_tile] #where is interface relative to new tile
	return (pos_new[0], pos_new[1], contact_point, interface_type)




############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
	assert interfaces_bind(1, 2) and interfaces_bind(3, 4) and interfaces_bind(2, 1) and not interfaces_bind(3, 2)








