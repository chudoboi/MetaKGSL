import pandas as pd
import os
import re
import shutil

data_dir = 'mod'
trans_save_dir = 'new_mod'
data_lists = ['train.txt', 'test.txt', 'valid.txt']
data_id_lists = ['entity2id.txt', 'relation2id.txt']

entity_dict_file = data_id_lists[0]
relation_dict_file = data_id_lists[1]
print('-----Loading entity dict-----')
entity_df = pd.read_table(os.path.join(data_dir, entity_dict_file), header=None)
entity_dict = dict(zip(entity_df[0], entity_df[1]))
print('-----Loading relation dict-----')
relation_df = pd.read_table(os.path.join(data_dir, relation_dict_file), header=None)
relation_dict = dict(zip(relation_df[0], relation_df[1]))

for data_list in data_lists:
    name = re.match(r'^(.*?)\.txt$', data_list).group(1)
    df_triples = []
    df = pd.read_table(os.path.join(data_dir, data_list), header=None)
    df[[1, 2]] = df[[2, 1]]
    df_triples = list(zip([entity_dict[h] for h in df[0]],
                                         [entity_dict[t] for t in df[1]],
                                         [relation_dict[r] for r in df[2]]))
    save_path = os.path.join(trans_save_dir, name+'2id.txt')
    with open(save_path, 'w') as file:
        file.write(f"{len(df_triples)}\n")
        for item in df_triples:
            file.write(' '.join(map(str, item)) + '\n')

for data_id_list in data_id_lists:
    input_file = os.path.join(data_dir, data_id_list)
    with open(input_file, 'r') as file:
        lines = file.readlines()
    data_line_count = len(lines)
    output_file = os.path.join(trans_save_dir, data_id_list)
    with open(output_file, 'w') as file:
        file.write(f"{data_line_count}\n") 
        file.writelines(lines)
        