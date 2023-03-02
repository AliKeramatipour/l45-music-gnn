import torch

def create_tonnetz_adjacency_matrix(num_notes):
    # In Tonnetz, each node has six neighbours which have pitches of the following distances (in semi-tones)
    # E.g. C4 has neighbours F3, G#3, A3, D#4, E4, G4
    NEIGHBOUR_DISTANCES = [-7, -4, -3, 3, 4, 7]
    A = []

    for i in range(num_notes):
        row = torch.zeros(num_notes, dtype=torch.int)
        for d in NEIGHBOUR_DISTANCES:
            j = i+d
            if j >= 0 and j < num_notes:
                row[j] = 1
        A.append(row)
    return torch.stack(A)

def get_tonnetz_edge_index(num_notes):
    adj_mat = create_tonnetz_adjacency_matrix(num_notes)
    edge_index = adj_mat.to_sparse().indices()
    return edge_index


# function which takes music sequence and window size(history length) and return training data.
# music = Time_step x Num_nodes
def slice_temporal_data(music_seq,window_size=5):
    return [[torch.transpose(music_seq[i:i+window_size],0,1),music_seq[i+window_size].reshape(-1,1)] for i in range(len(music_seq)-window_size)]