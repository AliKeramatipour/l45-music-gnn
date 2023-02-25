# based on Ali's codes
# modified based on https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c
import mido
import numpy as np
import torch

# track number is set to be 1 assuming the track01 is for piano
def get_midi_timesteps(midi,track_num=1):
    track = midi.tracks[track_num]
    return sum([m.time for m in track])

def midi_to_tensor(midi,track_num=1):
    # Extract information about the notes being played
    max_timesteps = get_midi_timesteps(midi,track_num=track_num)
    tensor = np.zeros((max_timesteps, 128))
    previous_note = [0] * 128
    timesteps = 0
    track = midi.tracks[track_num]
    for msg in track:
        timesteps += msg.time
        if msg.type == 'note_on':
            tmp = previous_note[msg.note]
            tensor[tmp:timesteps, msg.note] = tensor[tmp, msg.note]
            tensor[timesteps, msg.note] = msg.velocity
            previous_note[msg.note] = timesteps
        if msg.type == 'note_off':
            tmp = previous_note[msg.note]
            tensor[tmp:timesteps, msg.note] = tensor[tmp, msg.note]
            tensor[timesteps, msg.note] = 0
            previous_note[msg.note] = timesteps
    return torch.from_numpy(tensor)

def tensor_to_midi(tensor, filename):
    # Create a MIDI file
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    # Convert the tensor into messages
    current_timestep = 0
    for timestep in range(1,tensor.shape[0]):
        for note in range(0,128):
            if tensor[timestep, note] - tensor[timestep - 1, note] != 0:
                velocity = int(tensor[timestep, note])
                msg = mido.Message('note_on', note=note, velocity=velocity, time=timestep-current_timestep)
                track.append(msg)
                current_timestep = timestep
    # Save the MIDI file
    mid.save(filename)

if __name__ == '__main__':
    from pathlib import Path
    import os
    root = Path(__file__).parent.parent.absolute()
    data_path = os.path.join(root,"music_data","src")
    output_path = os.path.join(root,"music_data","output")
    data = os.path.join(data_path,"example.mid")
    output_data = os.path.join(output_path,"example_out.mid")
    print(data)
    print(output_data)

    midi = mido.MidiFile(data)
    midi_tensor = midi_to_tensor(midi)
    print("MIDI -> Tensor: completed")
    #tensor_to_midi(midi_tensor,output_data)
    #midi.tracks.pop(0)
    #midi.save(output_data)
    print("Tensor -> MIDI: completed")