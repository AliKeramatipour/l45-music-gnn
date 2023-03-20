import torch

# In Tonnetz, each node has six neighbours which have pitches of the following distances (in semi-tones)
# E.g. C4 has neighbours F3, G#3, A3, D#4, E4, G4
NEIGHBOUR_DISTANCES = [-7, -4, -3, 3, 4, 7]
    
def create_tonnetz_adjacency_matrix(num_notes):
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
    
def get_edge_attributes(A):
  # Returns edge attributes for the adjacency matrix `A`, 
  # where the edge attributes are a one-hot encoding for each type of edge
  edge_index = A.to_sparse().indices()
  edge_attr_raw = []
  for i in range(edge_index.shape[1]):
    distance = (edge_index[1][i] - edge_index[0][i]).item()
    edge_attr_raw.append(NEIGHBOUR_DISTANCES.index(distance))
  return F.one_hot(torch.tensor(edge_attr_raw))


# function which takes music sequence and window size(history length) and return training data.
# music = Time_step x Num_nodes
def slice_temporal_data(music_seq,window_size=5):
    return [[torch.transpose(music_seq[i:i+window_size],0,1),music_seq[i+window_size].reshape(-1,1)] for i in range(len(music_seq)-window_size)]
    
    
def calculate_compress_factor(file_name, desired_tpb):
  mid = mido.MidiFile(file_name)
  compress_factor = mid.ticks_per_beat / desired_tpb
  if compress_factor % 1.0 != 0.0:
    logging.warning(f"compress_factor of {compress_factor} is not an integer, rounding up...")
  compress_factor = math.ceil(compress_factor)
  return compress_factor

def compress_tensor(tensor, file_name, method, desired_tpb=16):
  '''
  Reduces the fidelity of the musical tensor, i.e. merge multiple timesteps into one step

  Args:
    `tensor`: PyTorch tensor of shape (timesteps, num_notes)
    `file_name`: path to the MIDI file
    `method`: str in ["max", "avg", "majority"]
    `desired_tpb`: desired ticks per beat for the tensor to be compressed to
  '''
  tensor_np = tensor.cpu().detach().cpu()
  assert(len(tensor_np.shape) == 2)

  compress_factor = calculate_compress_factor(file_name, desired_tpb)
  compressed_vectors = []
  length = tensor_np.shape[0]
  for start in range(0, length, compress_factor):
    end = min(start + compress_factor, length)
    tensor_slice = tensor_np[start:end, :]
    if (method == "max"):
      raise NotImplementedError()
    elif (method == "avg"):
      raise NotImplementedError()
    elif (method == "majority"):
      majority = (end-start) / 2
      majority_nonzeroes = np.count_nonzero(tensor_slice, axis=0) >= majority
      compressed_vectors.append((majority_nonzeroes).astype(int))
    else:
      raise KeyError(f"Unknown method {method}")
  return torch.tensor(compressed_vectors)

def reduce_tensor(tensor, start_note, end_note):
  '''
  Args:
    `tensor`: PyTorch tensor of shape (timesteps, num_notes)
    `start_note`: note to start the tensor from (integer in 0-127)
    `end_note`: note to end the tensor at (integer in 0-127)
  '''
  assert(end_note >= start_note)
  return tensor[:, start_note:end_note+1]

def uncompress_tensor(tensor, orig_tpb, compressed_tpb):
  '''
  Args:
    `tensor`: PyTorch tensor of shape (timesteps, num_notes)
    `orig_tpb`: ticks per beat of the original/generated MIDI file
    `compressed_tpb`: ticks per beat used by the compressed `tensor`
  '''
  compress_factor = orig_tpb / compressed_tpb
  if compress_factor % 1.0 != 0.0:
    logging.warning(f"compress_factor of {compress_factor} is not an integer, rounding up...")
  compress_factor = math.ceil(compress_factor)
  # "Stretch" out the tensor using Kronecker product
  return torch.kron(tensor, torch.ones((compress_factor, 1)))

def unreduce_tensor(tensor, start_note, end_note):
  '''
  Expands out a reduced tensor to include all 128 notes in the MIDI range

  Args:
    `tensor`: PyTorch tensor of shape (timesteps, num_notes)
    `start_note`: MIDI note that `tensor` starts from (integer in 0-127)
    `end_note`: MIDI note that `tensor` ends at (integer in 0-127)
  '''
  assert(end_note >= start_note)
  timesteps = tensor.shape[0]
  low_notes = torch.zeros((timesteps, start_note))
  high_notes = torch.zeros((timesteps, 127-end_note))
  return torch.cat((low_notes, tensor, high_notes), dim=1)