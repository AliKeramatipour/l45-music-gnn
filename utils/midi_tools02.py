# modified based on https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c
import mido
import numpy as np
import torch
from piano_roll_visualizer import piano_roll

# track number is set to be 1 assuming the track01 is for piano
def get_midi_timesteps(midi,track_num=1):
    track = midi.tracks[track_num]
    return sum([m.time for m in track])

def valid_msg(msg):
    return msg.type=="note_on" or msg.type=="note_off"

def get_new_state(new_msg, previous_state):
    new_state = [0] * 128 if previous_state is None else previous_state.copy()
    if valid_msg(new_msg):
        new_state[new_msg.note] = new_msg.velocity if new_msg.type=="note_on" else 0
    else:
        new_state = previous_state
    return (new_state, new_msg.time)

def track2matrix(track):
    result = []
    last_state =  [0]*128
    for i in range(len(track)):
        new_state, wait_time = get_new_state(track[i], last_state)
        if wait_time > 0:result += [last_state]*wait_time
        last_state, last_time = new_state, wait_time
    return result

def mid2arry(mid, track_num=1, min_msg_pct=0.1):
    mat = track2matrix(mid.tracks[track_num])
    # make all nested list the same length
    all_arys = np.array(mat)
    # trim: remove consecutive 0s in the beginning and at the end
    sums = all_arys.sum(axis=1)
    ends = np.where(sums > 0)[0]
    return all_arys[min(ends): max(ends)]

def arry2mid(ary, tempo=500000):
    new_ary = np.concatenate([np.array([[0] * 128]), np.array(ary)], axis=0)
    mid_new = mido.MidiFile()
    track = mido.MidiTrack()
    mid_new.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    changes = new_ary[1:] - new_ary[:-1]
    last_time = 0
    for ch in changes:
        if sum(ch) == 0:  # no change
            last_time += 1
        else:
            on_notes = np.where(ch > 0)[0]
            on_notes_velocity = ch[on_notes]
            off_notes = np.where(ch < 0)[0]
            first_ = True
            for n, v in zip(on_notes, on_notes_velocity):
                new_time = last_time if first_ else 0
                track.append(mido.Message('note_on', note=int(n), velocity=int(v), time=int(new_time)))
                first_ = False
            for n in off_notes:
                new_time = last_time if first_ else 0
                track.append(mido.Message('note_off', note=int(n), velocity=0, time=int(new_time)))
                first_ = False
            last_time = 0
    return mid_new


if __name__ == '__main__':
    from pathlib import Path
    import os
    import time
    root = Path(__file__).parent.parent.absolute()
    data_path = os.path.join(root,"music_data","src")
    output_path = os.path.join(root,"music_data","output")
    data = os.path.join(data_path,"example.mid")
    output_data = os.path.join(output_path,"example_out.mid")
    print(data)
    print(output_data)

    midi = mido.MidiFile(data)
    arr = mid2arry(midi)
    #piano_roll(arr,plot_range=(arr.shape[0]-15000,arr.shape[0]))

    from midi_tools01 import midi_to_tensor_multi_tracks, tensor_to_midi
    tensor = midi_to_tensor_multi_tracks(midi)
    arr = tensor.cpu().detach().numpy()
    #piano_roll(arr,plot_range=(arr.shape[0]-15000,arr.shape[0]))
    print("Converting from array to MIDI")
    start = time.time()
    #new_midi = arry2mid(arr)
    new_midi = tensor_to_midi(tensor,output_data)
    print("Execution time:",time.time()-start)
    #new_midi.save(output_data)