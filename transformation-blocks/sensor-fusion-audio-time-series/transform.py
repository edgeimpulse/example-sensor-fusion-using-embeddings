import os
import sys
import argparse
import pandas as pd
import json
import librosa
from moviepy.editor import AudioFileClip
import numpy as np

def save_metadata(metadata, out_directory):
    all_ok = all(metadata['metadata'][m] != 0 for m in metadata['metadata'])
    metadata['metadata']['ei_check'] = 1 if all_ok else 0
    with open(os.path.join(out_directory, 'ei-metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    exit(0)

def extract_and_resample_audio(audio_file_path):
    """Load audio from MP4, extract it, and resample to 8kHz."""
    if audio_file_path.endswith('.mp4'):
        audio_clip = AudioFileClip(audio_file_path)
        audio_clip.write_audiofile("temp_audio.wav")
        y, sr = librosa.load("temp_audio.wav", sr=None)
        os.remove("temp_audio.wav")
    else:
        y, sr = librosa.load(audio_file_path, sr=None)
    
    y_resampled = librosa.core.resample(y, orig_sr=sr, target_sr=8000)
    return y_resampled

def interpolate_csv_to_frequency(csv_data, desired_frequency=8000):
    """Interpolate the CSV data to match the desired frequency."""
    # Calculate the total duration from the 'time' column
    start_time = csv_data['time'].iloc[0]
    end_time = csv_data['time'].iloc[-1]
    total_duration_seconds = (end_time - start_time) * 1e-9  # Convert nanoseconds to seconds

    # Calculate number of samples based on total duration and desired frequency
    num_samples = int(total_duration_seconds * desired_frequency)

    # Create new evenly spaced timestamps
    new_timestamps = np.linspace(start_time, end_time, num_samples)

    # Create a new dataframe with the new_timestamps
    interpolated_df = pd.DataFrame({"time": new_timestamps})

    # Linearly interpolate original CSV data based on new_timestamps
    for col in csv_data.columns:
        if col != "time":
            interpolated_series = np.interp(new_timestamps, csv_data['time'], csv_data[col])
            interpolated_df[col] = interpolated_series

    return interpolated_df


def main():
    parser = argparse.ArgumentParser(description='Organization transformation block')
    parser.add_argument('--in-directory', type=str, required=True, help="Input directory where your files are located")
    parser.add_argument('--out-directory', type=str, required=True, help="Output directory")
    parser.add_argument('--audio', type=str, required=True, help="Path to the audio file to be merged")
    parser.add_argument('--csv', type=str, required=True, help="CSV file")
    parser.add_argument('--metadata', type=json.loads, required=False, help="Existing metadata")
    
    args, unknown = parser.parse_known_args()

    # Load and process the audio data
    audio_data = extract_and_resample_audio(os.path.join(args.in_directory, args.audio))
    audio_df = pd.DataFrame(audio_data, columns=["audio"])

    # Load and interpolate the CSV data
    csv_data = pd.read_csv(os.path.join(args.in_directory, args.csv))
    print(csv_data.head(10))
    print(len(csv_data))

    interpolated_csv = interpolate_csv_to_frequency(csv_data)
    print(interpolated_csv.head(10))
    print(len(interpolated_csv))
    
    print(audio_df.head(10))
    print(len(audio_df))
    
    # Ensure the audio and CSV data have the same length
    min_len = min(len(audio_df), len(interpolated_csv))
    audio_df = audio_df.head(min_len)
    interpolated_csv = interpolated_csv.head(min_len)

    # Merge audio data with CSV data
    final_merged_data = pd.concat([interpolated_csv, audio_df], axis=1)
    initial_time = final_merged_data['time'].iloc[0]
    final_merged_data['time'] = (final_merged_data['time'] - initial_time) / 1e6
    # final_merged_data = final_merged_data.sort_values(by='time')
    print(final_merged_data.head(10))
    print(len(final_merged_data))

    # Save merged data to the out-directory
    output_path = os.path.join(args.out_directory, "merged_data.csv")
    final_merged_data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    main()