import torch
import os
import numpy as np
from tqdm import tqdm


def read_entity_from_id(filename):
    entity2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                entity, entity_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                entity2id[entity] = int(entity_id)
    return entity2id


def read_relation_from_id(filename):
    relation2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                relation, relation_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                relation2id[relation] = int(relation_id)
    return relation2id

def init_embeddings(entity_file, relation_file):
    entity_emb, relation_emb = [], []

    with open(entity_file) as f:
        for line in f:
            entity_emb.append([float(val) for val in line.strip().split()])

    with open(relation_file) as f:
        for line in f:
            relation_emb.append([float(val) for val in line.strip().split()])

    return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)

def parse_line(line):
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2

def load_data(filename, entity2id, relation2id, is_unweigted=False, directed=True):
    with open(filename) as f:
        lines = f.readlines()

    # this is list for relation triples
    triples_data = []
    rows, cols, data = [], [], []
    unique_entities = set()
    for line in tqdm(lines):
        e1, relation, e2 = parse_line(line)
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append(
            (entity2id[e1], relation2id[relation], entity2id[e2]))
        if not directed:
            rows.append(entity2id[e1])
            cols.append(entity2id[e2])
            if is_unweigted:
                data.append(1)
            else:
                data.append(relation2id[relation])

        rows.append(entity2id[e2])
        cols.append(entity2id[e1])
        if is_unweigted:
            data.append(1)
        else:
            data.append(relation2id[relation])
    return triples_data, (rows, cols, data), list(unique_entities)


def build_data(path, is_unweigted=False, directed=True):
    entity2id = read_entity_from_id(path + 'entity2id.txt')
    relation2id = read_relation_from_id(path + 'relation2id.txt')
    train_triples, train_adjacency_mat, unique_entities_train = load_data(os.path.join(
        path, 'train.txt'), entity2id, relation2id, is_unweigted, directed)
    validation_triples, valid_adjacency_mat, unique_entities_validation = load_data(
        os.path.join(path, 'valid.txt'), entity2id, relation2id, is_unweigted, directed)
    test_triples, test_adjacency_mat, unique_entities_test = load_data(os.path.join(
        path, 'test.txt'), entity2id, relation2id, is_unweigted, directed)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    left_entity, right_entity = {}, {}

    with open(os.path.join(path, 'train.txt')) as f:
        lines = f.readlines()

    for line in tqdm(lines):
        e1, relation, e2 = parse_line(line)

        # Count number of occurences for each (e1, relation)
        if relation2id[relation] not in left_entity:
            left_entity[relation2id[relation]] = {}
        if entity2id[e1] not in left_entity[relation2id[relation]]:
            left_entity[relation2id[relation]][entity2id[e1]] = 0
        left_entity[relation2id[relation]][entity2id[e1]] += 1

        # Count number of occurences for each (relation, e2)
        if relation2id[relation] not in right_entity:
            right_entity[relation2id[relation]] = {}
        if entity2id[e2] not in right_entity[relation2id[relation]]:
            right_entity[relation2id[relation]][entity2id[e2]] = 0
        right_entity[relation2id[relation]][entity2id[e2]] += 1

    left_entity_avg = {}
    for i in tqdm(range(len(relation2id))):
        left_entity_avg[i] = sum(
            left_entity[i].values()) * 1.0 / len(left_entity[i])

    right_entity_avg = {}
    for i in tqdm(range(len(relation2id))):
        right_entity_avg[i] = sum(
            right_entity[i].values()) * 1.0 / len(right_entity[i])

    headTailSelector = {}
    for i in tqdm(range(len(relation2id))):
        headTailSelector[i] = 1000 * right_entity_avg[i] / \
            (right_entity_avg[i] + left_entity_avg[i])
    # headTailSelector will then be used for dynamic negative sample generation
    # (get_iteration_batch() function in utiles.py)

    return (train_triples, train_adjacency_mat), (validation_triples, valid_adjacency_mat), (test_triples, test_adjacency_mat), \
        entity2id, relation2id, headTailSelector, unique_entities_train
