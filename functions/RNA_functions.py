import networkx as nx
import string
from os.path import isfile
import json
import numpy as np
try:
   import RNA
except ImportError:
   import sys
   sys.path.insert(0,'/software/nmartin/el6.5/viennarna_install/lib/python3.10/site-packages')
   sys.path.insert(0,'/users/nmartin/nmartin/ViennaRNA/lib/python3.11/site-packages')
   sys.path.insert(0,'/Users/noramartin/ViennaRNA/lib/python3.13/site-packages')
   import RNA
print('VIENNARNA version', RNA.__version__)

index_to_base = {0: 'A', 1: 'C', 2: 'U', 3: 'G'}
base_to_number={'A':0, 'C':1, 'U':2, 'G':3, 'T':2}

RNA.cvar.uniq_ML = 1

model = RNA.md()
#model.noLP = 1 # no isolate base pairs
model.pf_smooth = 0 # deactivate partition function smoothing



############################################################################################################
## manipulate structures
############################################################################################################

def generate_all_allowed_dotbracket(L, allow_isolated_bps=False, filename=None):
   """recursively build up all possible structures of length L;
    - allow_isolated_bps (True/False) to control whether isolated base pairs are permitted"""
   assert L <= 35 # avoid overflow errors
   if isfile(filename):
      with open(filename, 'r') as f:
         potential_structures = json.load(f) 
         return potential_structures
   potential_structures = [s for s in return_all_allowed_structures_starting_with('', L, allow_isolated_bps=allow_isolated_bps) if is_likely_to_be_valid_structure(s)]
   with open(filename, 'w') as f:
       json.dump(potential_structures, f) 
   assert len(set(potential_structures)) == len(potential_structures)
   return potential_structures

def return_all_allowed_structures_starting_with(structure, L, allow_isolated_bps=False):
   """recursively build up all possible structures of length L;
   already has some constraints of dot-bracket notations built in:
   only one closed bracket for each upstream open bracket, all brackets closed by the end, length of hairpin loops,
   optionally isolated base pairs"""
   if len(structure) == L:
      return [structure]
   assert len(structure) < L
   assert L-len(structure) <= 35
   structure_list = []
   if (len(structure) >= 2 and structure[-1] == '(' and structure[-2] != '(' and not allow_isolated_bps):
      structure_list += return_all_allowed_structures_starting_with(structure+'(', L, allow_isolated_bps) #need to add ( to avoid isolated bp
   elif (len(structure) == 1 and structure[-1] == '(' and not allow_isolated_bps):
      structure_list += return_all_allowed_structures_starting_with(structure+'(', L, allow_isolated_bps) #need to add ( to avoid isolated bp
   elif (len(structure) >= 2 and structure[-1] == ')' and structure[-2] != ')' and not allow_isolated_bps):
      structure_list += return_all_allowed_structures_starting_with(structure+')', L, allow_isolated_bps) #need to add ) to avoid isolated bp
   elif len(structure) <= 4: #at the beginning of a structure can only have opening brackets/loops
      structure_list += return_all_allowed_structures_starting_with(structure+'.', L, allow_isolated_bps) 
      structure_list += return_all_allowed_structures_starting_with(structure+'(', L, allow_isolated_bps) 
      if len(structure) == 4 and '(' in structure and '(' not in structure[-3:]:
         structure_list += return_all_allowed_structures_starting_with(structure+')', L, allow_isolated_bps)
   elif structure.count('(') > structure.count(')')+L-len(structure):
      pass  #cannot close all base pairs anymore, return empty list
   elif structure.count('(') == structure.count(')')+L-len(structure) and '(' not in structure[-3:]: #need to close base pairs
      structure_list += return_all_allowed_structures_starting_with(structure+')', L, allow_isolated_bps)    
   elif structure.count('(') > structure.count(')') and allow_isolated_bps and '(' not in structure[-3:]: #upstream bps are open, so closing also allowed
      structure_list += return_all_allowed_structures_starting_with(structure+'.', L, allow_isolated_bps)
      if L-len(structure) > 3:
         structure_list += return_all_allowed_structures_starting_with(structure+'(', L, allow_isolated_bps)
      structure_list += return_all_allowed_structures_starting_with(structure+')', L, allow_isolated_bps)
   elif structure.count('(') > structure.count(')') and structure[-1] == ')' and not allow_isolated_bps: # everything allowed
      structure_list += return_all_allowed_structures_starting_with(structure+'.', L, allow_isolated_bps)
      if L-len(structure) > 3:
         structure_list += return_all_allowed_structures_starting_with(structure+'(', L, allow_isolated_bps)
      structure_list += return_all_allowed_structures_starting_with(structure+')', L, allow_isolated_bps)
   elif structure.count('(') > structure.count(')')+1 and not allow_isolated_bps and '(' not in structure[-3:]: #at least two upstream bps are open, so closing also allowed with non-isolated bps
      structure_list += return_all_allowed_structures_starting_with(structure+'.', L, allow_isolated_bps)
      if L-len(structure) > 3:
         structure_list += return_all_allowed_structures_starting_with(structure+'(', L, allow_isolated_bps)
      structure_list += return_all_allowed_structures_starting_with(structure+')', L, allow_isolated_bps)
   else: # can open new base pairs or introduce loop
      structure_list += return_all_allowed_structures_starting_with(structure+'.', L, allow_isolated_bps)
      if L-len(structure) > 3:
         structure_list += return_all_allowed_structures_starting_with(structure+'(', L, allow_isolated_bps)
   return structure_list
############################################################################################################
## check viability of structure
############################################################################################################

def hairpin_loops_long_enough(structure):
   """check if any paired sites in the dot-bracket input structure
   are at least four sites apart"""
   bp_mapping = get_basepair_indices_from_dotbracket(structure)
   for baseindex1, baseindex2 in bp_mapping.items():
      if abs(baseindex2-baseindex1) < 4:
         return False
   return True



def is_likely_to_be_valid_structure(structure):
   """tests if a structure in dotbracket format is likely to be a valid structure:
   basepairs closed, length of hairpin loops (>3), presence of basepairs and optionally isolated base pairs"""
   if not basepairs_closed(structure):
      return False
   if not hairpin_loops_long_enough(structure):
      return False
   if not structure.count(')') > 0:
      return False
   else:
      return True


def basepairs_closed(structure):
   """test if all brackets are closed correctly in a dot-bracket string"""
   try:
      bp_map = get_basepair_indices_from_dotbracket(structure)
      return True
   except (ValueError, KeyError):
      return False

def sequence_compatible_with_basepairs(sequence, structure):
   """check if the input sequence (string containing AUGC) is 
   compatibale with the dot-bracket input structure,
   i.e. paired sites are a Watson-Crick pair or GU"""
   allowed_basepairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']
   for b in sequence:
      assert b in ['A', 'U', 'G', 'C']
   bp_mapping = get_basepair_indices_from_dotbracket(structure)
   for baseindex1, baseindex2 in bp_mapping.items():
      if sequence[baseindex1]+sequence[baseindex2] not in allowed_basepairs:
         return False
   return True


def get_basepair_indices_from_dotbracket(dotbracketstring):
   """extract a dictionary mapping each paired position with its partner:
   each base pair is represented twice: mapping from opening to closing bracket and vice versa"""
   base_pair_mapping = {}
   number_open_brackets = 0
   opening_level_vs_index = {}
   for charindex, char in enumerate(dotbracketstring):
      if char in ['(', '[']:
         number_open_brackets += 1
         opening_level_vs_index[number_open_brackets] = charindex
      elif char in [')', ']']:
         base_pair_mapping[charindex] = opening_level_vs_index[number_open_brackets]
         base_pair_mapping[opening_level_vs_index[number_open_brackets]] = charindex
         del opening_level_vs_index[number_open_brackets]
         number_open_brackets -= 1
      elif char in ['.', '_']:
         pass
      else:
         raise ValueError('invalid character in dot-bracket string')
      if number_open_brackets < 0:
         raise ValueError('invalid dot-bracket string')
   if number_open_brackets != 0:
      raise ValueError('invalid dot-bracket string')
   return base_pair_mapping
###############################################################################################
## converting between sequence representations
###############################################################################################
def sequence_int_to_str(sequence_indices_tuple):
   """convert between sequence representations:
   from tuple of integers from 0-3 to biological four-letter string;
   motive: these integers can be used as array tuples"""
   return ''.join([index_to_base[ind] for ind in sequence_indices_tuple])

def sequence_str_to_int(sequence_str):
   """convert between sequence representations:
   from tuple of integers from 0-3 to biological four-letter string;
   motive: these integers can be used as array tuples"""
   return tuple([base_to_number[c] for c in sequence_str])
###############################################################################################
## focus on minimum-free energy structure and energy: 
## not tested whether there is an energy gap between top two structures
###############################################################################################
def get_all_mfe_structures_seq_str(seq_tuple, db_to_int, temp=37):
   """get minimum free energy structure in dotbracket format for the sequence in integer format"""
   subopt = return_structures_in_energy_range(0.02, sequence_int_to_str(seq_tuple), temp=temp)
   return [db_to_int[s] for s in subopt.keys()]

def return_structures_in_energy_range(G_range_kcal, seq, temp=37):
   """ return structures in energy range (kcal/mol) of G_range_kcal from the mfe of sequence sequence_indices;
   code adapted from ViennaRNA documentation: https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/RNAlib-2.4.14.pdf;
   subopt is run for a small energy range at first and then a larger one, until a suboptimal structure is identified"""
   RNA.cvar.uniq_ML, structure_vs_energy = 1, {} # Set global switch for unique multiloop decomposition
   model.temperature = temp
   fold_compound_seq = RNA.fold_compound(seq, model)
   (mfe_structure, mfe) = fold_compound_seq.mfe()
   fold_compound_seq.subopt_cb(int(G_range_kcal*100.0*1.1), convert_result_to_dict, structure_vs_energy)
   subopt_structure_vs_G = {alternativestructure: fold_compound_seq.eval_structure(alternativestructure) for alternativestructure in structure_vs_energy 
                               if abs(fold_compound_seq.eval_structure(alternativestructure)-mfe) <= G_range_kcal}
   subopt_structure_vs_G[mfe_structure] = mfe
   return subopt_structure_vs_G

def convert_result_to_dict(structure, energy, data):
   """ function needed for subopt from ViennaRNA documentation: https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/RNAlib-2.4.14.pdf"""
   if not structure == None:
      data[structure] = energy


def get_Boltzmann_ensemble(seq_tuple, db_to_structure_int, temp=37):
   assert '.'*len(seq_tuple) in db_to_structure_int
   kbT_RNA = 0.001987204259 * (temp + 273) #kcal/mol
   RNA.cvar.uniq_ML = 1 # Set global switch for unique multiloop decomposition
   model.temperature = temp
   seq_str = sequence_int_to_str(seq_tuple)
   a = RNA.fold_compound(seq_str, model)
   (mfe_structure, mfe) = a.mfe()
   structure_vs_energy = {i: a.eval_structure(db) for db, i in db_to_structure_int.items() if sequence_compatible_with_basepairs(seq_str, db)}
   assert db_to_structure_int[mfe_structure] in structure_vs_energy
   structure_list = [s for s in structure_vs_energy.keys()]
   weight_list = np.exp(np.array([structure_vs_energy[s] for s in structure_list]) * -1.0/kbT_RNA)
   Z = np.sum(weight_list)
   del a, mfe_structure, mfe, structure_vs_energy
   return {s: w/Z for s, w in zip(structure_list, weight_list)}
############################################################################################################
## test
############################################################################################################
def get_Boltzmann_ensemble_fixed_cutoff(seq_tuple, db_to_structure_int, cutoff_in_kT = 1000, temp=37):
   kbT_RNA = 0.001987204259 * (temp + 273) #kcal/mol
   seq_str, cutoff_kcal = sequence_int_to_str(seq_tuple), cutoff_in_kT * kbT_RNA
   #relevant_structures = {s: db_to_structure_int[s] for s in return_structures_in_energy_range(cutoff_in_kT*kbT_RNA, seq_str).keys()}
   RNA.cvar.uniq_ML, structure_vs_energy = 1, {} # Set global switch for unique multiloop decomposition
   model.temperature = temp
   model.pf_smooth = 0
   a = RNA.fold_compound(seq_str, model)
   (mfe_structure, mfe) = a.mfe()
   a.exp_params_rescale(mfe)
   (bp_propensity, Z_vienna_dG) = a.pf()
   #a.exp_params_rescale(mfe)
   a.subopt_cb(int(cutoff_kcal*100.0*1.1), convert_result_to_dict, structure_vs_energy)
   #subopt_structure_vs_G = {s2: a.eval_structure(s2) for s2 in structure_vs_energy}  ######need to test that this is correct
   subopt_structure_vs_G_checked = {s2: G for s2, G in structure_vs_energy.items() if abs(G - mfe) < cutoff_kcal}
   assert mfe_structure in structure_vs_energy
   weight_list = np.exp(np.array([G for G in structure_vs_energy.values()]) * -1.0/kbT_RNA)
   if len(db_to_structure_int):
      structure_list = [db_to_structure_int[s] for s in structure_vs_energy.keys()]
   else:
      structure_list = [s for s in structure_vs_energy.keys()]
   Z = np.sum(weight_list)
   if not abs(Z/np.exp(Z_vienna_dG * (-1)/kbT_RNA) - 1) < 10**(-2):
      print(seq_str, Z, np.exp(Z_vienna_dG * (-1)/kbT_RNA), structure_list)
   assert abs(Z/np.exp(Z_vienna_dG * (-1)/kbT_RNA) - 1) < 10**(-3)
   del seq_str, a, mfe_structure, mfe, structure_vs_energy, subopt_structure_vs_G_checked
   return {s: w/Z for s, w in zip(structure_list, weight_list)}

if __name__ == "__main__":
   kbT_RNA = RNA.exp_param().kT/1000.0 ## from https://github.com/ViennaRNA/ViennaRNA/issues/58
   assert abs(kbT_RNA/(0.001987204259 * (37 + 273.15)) - 1) < 0.001 # value here and above from wikipedia https://en.wikipedia.org/wiki/Boltzmann_constant
   ####
   db_to_structure_int = {'.......(...)': 1, '......(....)': 2, '......(...).': 3, '.....(.....)': 4, '.....(....).': 5, '.....(...)..': 6, '.....((...))': 7, '....(......)': 8, '....(.....).': 9, '....(....)..': 10, '....(...)...': 11, '....(.(...))': 12, '....((....))': 13, '....((...).)': 14, '....((...)).': 15, '...(.......)': 16, '...(......).': 17, '...(.....)..': 18, '...(....)...': 19, '...(...)....': 20, '...(..(...))': 21, '...(.(....))': 22, '...(.(...).)': 23, '...(.(...)).': 24, '...((.....))': 25, '...((....).)': 26, '...((....)).': 27, '...((...)..)': 28, '...((...).).': 29, '...((...))..': 30, '...(((...)))': 31, '..(........)': 32, '..(.......).': 33, '..(......)..': 34, '..(.....)...': 35, '..(....)....': 36, '..(...(...))': 37, '..(...).....': 38, '..(...)(...)': 39, '..(..(....))': 40, '..(..(...).)': 41, '..(..(...)).': 42, '..(.(.....))': 43, '..(.(....).)': 44, '..(.(....)).': 45, '..(.(...)..)': 46, '..(.(...).).': 47, '..(.(...))..': 48, '..(.((...)))': 49, '..((......))': 50, '..((.....).)': 51, '..((.....)).': 52, '..((....)..)': 53, '..((....).).': 54, '..((....))..': 55, '..((...)...)': 56, '..((...)..).': 57, '..((...).)..': 58, '..((...))...': 59, '..((.(...)))': 60, '..(((....)))': 61, '..(((...).))': 62, '..(((...)).)': 63, '..(((...))).': 64, '.(.........)': 65, '.(........).': 66, '.(.......)..': 67, '.(......)...': 68, '.(.....)....': 69, '.(....(...))': 70, '.(....).....': 71, '.(....)(...)': 72, '.(...(....))': 73, '.(...(...).)': 74, '.(...(...)).': 75, '.(...)......': 76, '.(...).(...)': 77, '.(...)(....)': 78, '.(...)(...).': 79, '.(..(.....))': 80, '.(..(....).)': 81, '.(..(....)).': 82, '.(..(...)..)': 83, '.(..(...).).': 84, '.(..(...))..': 85, '.(..((...)))': 86, '.(.(......))': 87, '.(.(.....).)': 88, '.(.(.....)).': 89, '.(.(....)..)': 90, '.(.(....).).': 91, '.(.(....))..': 92, '.(.(...)...)': 93, '.(.(...)..).': 94, '.(.(...).)..': 95, '.(.(...))...': 96, '.(.(.(...)))': 97, '.(.((....)))': 98, '.(.((...).))': 99, '.(.((...)).)': 100, '.(.((...))).': 101, '.((.......))': 102, '.((......).)': 103, '.((......)).': 104, '.((.....)..)': 105, '.((.....).).': 106, '.((.....))..': 107, '.((....)...)': 108, '.((....)..).': 109, '.((....).)..': 110, '.((....))...': 111, '.((...)....)': 112, '.((...)...).': 113, '.((...)..)..': 114, '.((...).)...': 115, '.((...))....': 116, '.((..(...)))': 117, '.((.(....)))': 118, '.((.(...).))': 119, '.((.(...)).)': 120, '.((.(...))).': 121, '.(((.....)))': 122, '.(((....).))': 123, '.(((....)).)': 124, '.(((....))).': 125, '.(((...)..))': 126, '.(((...).).)': 127, '.(((...).)).': 128, '.(((...))..)': 129, '.(((...)).).': 130, '.(((...)))..': 131, '.((((...))))': 132, '(..........)': 133, '(.........).': 134, '(........)..': 135, '(.......)...': 136, '(......)....': 137, '(.....(...))': 138, '(.....).....': 139, '(.....)(...)': 140, '(....(....))': 141, '(....(...).)': 142, '(....(...)).': 143, '(....)......': 144, '(....).(...)': 145, '(....)(....)': 146, '(....)(...).': 147, '(...(.....))': 148, '(...(....).)': 149, '(...(....)).': 150, '(...(...)..)': 151, '(...(...).).': 152, '(...(...))..': 153, '(...((...)))': 154, '(...).......': 155, '(...)..(...)': 156, '(...).(....)': 157, '(...).(...).': 158, '(...)(.....)': 159, '(...)(....).': 160, '(...)(...)..': 161, '(...)((...))': 162, '(..(......))': 163, '(..(.....).)': 164, '(..(.....)).': 165, '(..(....)..)': 166, '(..(....).).': 167, '(..(....))..': 168, '(..(...)...)': 169, '(..(...)..).': 170, '(..(...).)..': 171, '(..(...))...': 172, '(..(.(...)))': 173, '(..((....)))': 174, '(..((...).))': 175, '(..((...)).)': 176, '(..((...))).': 177, '(.(.......))': 178, '(.(......).)': 179, '(.(......)).': 180, '(.(.....)..)': 181, '(.(.....).).': 182, '(.(.....))..': 183, '(.(....)...)': 184, '(.(....)..).': 185, '(.(....).)..': 186, '(.(....))...': 187, '(.(...)....)': 188, '(.(...)...).': 189, '(.(...)..)..': 190, '(.(...).)...': 191, '(.(...))....': 192, '(.(..(...)))': 193, '(.(.(....)))': 194, '(.(.(...).))': 195, '(.(.(...)).)': 196, '(.(.(...))).': 197, '(.((.....)))': 198, '(.((....).))': 199, '(.((....)).)': 200, '(.((....))).': 201, '(.((...)..))': 202, '(.((...).).)': 203, '(.((...).)).': 204, '(.((...))..)': 205, '(.((...)).).': 206, '(.((...)))..': 207, '(.(((...))))': 208, '((........))': 209, '((.......).)': 210, '((.......)).': 211, '((......)..)': 212, '((......).).': 213, '((......))..': 214, '((.....)...)': 215, '((.....)..).': 216, '((.....).)..': 217, '((.....))...': 218, '((....)....)': 219, '((....)...).': 220, '((....)..)..': 221, '((....).)...': 222, '((....))....': 223, '((...(...)))': 224, '((...).....)': 225, '((...)....).': 226, '((...)...)..': 227, '((...)..)...': 228, '((...).)....': 229, '((...)(...))': 230, '((...)).....': 231, '((...))(...)': 232, '((..(....)))': 233, '((..(...).))': 234, '((..(...)).)': 235, '((..(...))).': 236, '((.(.....)))': 237, '((.(....).))': 238, '((.(....)).)': 239, '((.(....))).': 240, '((.(...)..))': 241, '((.(...).).)': 242, '((.(...).)).': 243, '((.(...))..)': 244, '((.(...)).).': 245, '((.(...)))..': 246, '((.((...))))': 247, '(((......)))': 248, '(((.....).))': 249, '(((.....)).)': 250, '(((.....))).': 251, '(((....)..))': 252, '(((....).).)': 253, '(((....).)).': 254, '(((....))..)': 255, '(((....)).).': 256, '(((....)))..': 257, '(((...)...))': 258, '(((...)..).)': 259, '(((...)..)).': 260, '(((...).)..)': 261, '(((...).).).': 262, '(((...).))..': 263, '(((...))...)': 264, '(((...))..).': 265, '(((...)).)..': 266, '(((...)))...': 267, '(((.(...))))': 268, '((((....))))': 269, '((((...).)))': 270, '((((...)).))': 271, '((((...))).)': 272, '((((...)))).': 273, '............': 0}
   structure_int_to_db = {i: d for d, i in db_to_structure_int.items()}
   #seq_tuple = sequence_str_to_int('CCUAGCUUGGGU')
   for temp in [30, 37, 44]:
      for repetition in range(10**3 + 1):
         if repetition == 10**3:
            seq_tuple = sequence_str_to_int('CCUAGCUUGGGU')
         else:
             seq_tuple = tuple(np.random.choice(4, size=12, replace=True))
         ensemble1 = get_Boltzmann_ensemble_fixed_cutoff(seq_tuple, db_to_structure_int, cutoff_in_kT = 1000, temp=37)
         ensemble2 = get_Boltzmann_ensemble(seq_tuple, db_to_structure_int, temp=37)
         for s in ensemble1:
            assert s in ensemble2 and abs(ensemble1[s] - ensemble2[s]) < 10**(-3)
         for s in ensemble2:
            assert s in ensemble1 and abs(ensemble1[s] - ensemble2[s]) < 10**(-3)
         if repetition >= 10**3 - 10 and temp==37:
            print(sequence_int_to_str(seq_tuple))
            print(sorted([(P, structure_int_to_db[s]) for s, P in ensemble1.items() if P > 0.01]), '\n')
            print(sorted([(P, structure_int_to_db[s]) for s, P in ensemble2.items() if P > 0.01]), '\n')
