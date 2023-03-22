import matplotlib.pyplot as plt
import numpy as np
import mido

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

def mid2arry(mid, track_num=0, min_msg_pct=0.1):
    mat = track2matrix(mid.tracks[track_num])
    # make all nested list the same length
    all_arys = np.array(mat)
    # trim: remove consecutive 0s in the beginning and at the end
    sums = all_arys.sum(axis=1)
    ends = np.where(sums > 0)[0]
    return all_arys[min(ends): max(ends)]

def convert2binary(arr):
    return np.where(arr>0,1,0)

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

def piano_roll(array,plot_range = (0,3000)):
    array = array[plot_range[0]:plot_range[1]]
    plt.plot(range(array.shape[0]), np.multiply(np.where(array>0, 1, 0), range(1, 129)), marker='.', markersize=1, linestyle='')
    plt.title("Piano roll")
    plt.xlabel("Timessteps")
    plt.ylabel("Notes")
    plt.show()

if __name__ == '__main__':
    midi = mido.MidiFile("sample_music_0_RNN2GCN1_acc_0.123.midi")
    arr = mid2arry(midi)
    piano_roll(arr)

