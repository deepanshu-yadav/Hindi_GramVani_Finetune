import json
import os
import shutil
import wget
import tarfile
import random
from pathlib import Path
import soundfile as sf
import librosa
from tqdm import tqdm

# Define paths
base_dir = os.path.join(os.getcwd(), "data", "GigaVoice")
train_dir = os.path.join(base_dir, "GV_Train_100h")
val_dir = os.path.join(base_dir, "GV_Dev_5h")
test_dir = os.path.join(base_dir, "GV_Eval_3h")
train_audio_dir = os.path.join(train_dir, "Audio")
val_audio_dir = os.path.join(val_dir, "Audio")
test_audio_dir = os.path.join(test_dir, "Audio")
train_text_file = os.path.join(train_dir, "text")
val_text_file = os.path.join(val_dir, "text")
test_text_file = os.path.join(test_dir, "text")
output_audio_dir = os.path.join(base_dir, "wavs_16k")
train_manifest =  "train_manifest.json"
val_manifest =  "val_manifest.json"
test_manifest =  "test_manifest.json"

# Create directories
os.makedirs(base_dir, exist_ok=True)
os.makedirs(output_audio_dir, exist_ok=True)

# Dataset URLs and local paths
dataset_urls = {
    "train": "https://www.openslr.org/resources/118/GV_Train_100h.tar.gz",
    "val": "https://www.openslr.org/resources/118/GV_Dev_5h.tar.gz",
    "test": "https://www.openslr.org/resources/118/GV_Eval_3h.tar.gz"
}
dataset_tars = {
    "train": os.path.join(base_dir, "GV_Train_100h.tar.gz"),
    "val": os.path.join(base_dir, "GV_Dev_5h.tar.gz"),
    "test": os.path.join(base_dir, "GV_Eval_3h.tar.gz")
}
dataset_dirs = {
    "train": train_dir,
    "val": val_dir,
    "test": test_dir
}

# Target sample rate
target_sr = 16000

# Set random seed for reproducibility
random.seed(42)

# Step 1: Download and extract datasets if they don't exist
def download_and_extract():
    for split, url in dataset_urls.items():
        tar_path = dataset_tars[split]
        extract_dir = dataset_dirs[split]
        if not os.path.exists(extract_dir):
            if not os.path.exists(tar_path):
                print(f"Downloading {split} dataset from {url}...")
                wget.download(url, tar_path)
                print(f"\nExtracting {split} dataset...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(base_dir)
            print(f"{split.capitalize()} dataset extracted to {extract_dir}")
        else:
            print(f"{split.capitalize()} dataset already exists at {extract_dir}")

# Step 2: Resample audio to 16kHz if needed
def resample_audio(audio_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith((".mp3", ".wav"))]
    for audio_file in tqdm(audio_files, desc=f"Resampling {audio_dir}"):
        input_path = os.path.join(audio_dir, audio_file)
        # Save as WAV regardless of input format
        output_filename = os.path.splitext(audio_file)[0] + ".wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # Skip if already resampled
        if os.path.exists(output_path):
            continue
        
        try:
            audio, sr = librosa.load(input_path, sr=None)
            if sr != target_sr:
                audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sf.write(output_path, audio_resampled, target_sr, subtype='PCM_16')
            else:
                # Copy audio data directly
                audio_resampled = audio
                sf.write(output_path, audio_resampled, target_sr, subtype='PCM_16')
        except Exception as e:
            print(f"Error resampling {audio_file}: {e}")

# Step 3: Process audio and text files to create manifest entries
def process_files(audio_dir, text_path, output_audio_dir):
    entries = []
    
    if not os.path.exists(text_path):
        print(f"Text file {text_path} not found, skipping.")
        return entries
    if not os.path.exists(audio_dir):
        print(f"Audio directory {audio_dir} not found, skipping.")
        return entries
    
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines(), desc=f"Processing {text_path}"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    print(f"Skipping malformed line in {text_path}: {line}")
                    continue
                
                file_name = parts[0]
                text = parts[1]
                # Skip if text is empty or only whitespace
                if not text or text.isspace():
                    print(f"Skipping empty transcription for {file_name}")
                    continue
                
                audio_file = os.path.join(audio_dir, file_name)
                # Use WAV extension for resampled file
                new_audio_file = os.path.join(output_audio_dir, file_name.rsplit(".", 1)[0] + ".wav")
                
                if not os.path.exists(new_audio_file):
                    print(f"Resampled audio file {new_audio_file} not found, skipping.")
                    continue
                
                # Calculate duration
                try:
                    duration = librosa.get_duration(path=new_audio_file)
                except Exception as e:
                    print(f"Error calculating duration for {new_audio_file}: {e}")
                    continue
                
                # Normalize audio path
                normalized_audio_path = str(Path(new_audio_file).as_posix())
                
                entries.append({
                    "audio_filepath": normalized_audio_path,
                    "text": text,  # Keep original text with Hindi characters
                    "duration": duration
                })
        return entries
    except Exception as e:
        print(f"Error processing text file {text_path}: {e}")
        return entries

# Step 4: Write manifest file
def write_manifest(entries, filename):
    if not entries:
        print(f"No entries to write for {filename}")
        return
    with open(filename, "w", encoding="utf-8") as f:
        for entry in entries:
            json.dump(entry, f, ensure_ascii=False)  # Preserve Unicode characters
            f.write("\n")
    print(f"Created {filename} with {len(entries)} entries")

def clean_remaining_file():
    """
    Deletes downloaded tar files and extracted dataset directories while preserving
    the output audio directory and manifest files.
    """
    # Define paths from the script
    base_dir = os.path.join(os.getcwd(), "data", "GigaVoice")
    output_audio_dir = os.path.join(base_dir, "wavs_16k")
    dataset_tars = {
        "train": os.path.join(base_dir, "GV_Train_100h.tar.gz"),
        "val": os.path.join(base_dir, "GV_Dev_5h.tar.gz"),
        "test": os.path.join(base_dir, "GV_Eval_3h.tar.gz")
    }
    dataset_dirs = {
        "train": os.path.join(base_dir, "GV_Train_100h"),
        "val": os.path.join(base_dir, "GV_Dev_5h"),
        "test": os.path.join(base_dir, "GV_Eval_3h")
    }

    # Delete tar files
    for split, tar_path in dataset_tars.items():
        if os.path.exists(tar_path):
            try:
                os.remove(tar_path)
                print(f"Deleted tar file: {tar_path}")
            except Exception as e:
                print(f"Error deleting tar file {tar_path}: {e}")
        else:
            print(f"Tar file {tar_path} not found, skipping.")

    # Delete extracted directories
    for split, dir_path in dataset_dirs.items():
        if os.path.exists(dir_path) and dir_path != output_audio_dir:
            try:
                shutil.rmtree(dir_path)
                print(f"Deleted directory: {dir_path}")
            except Exception as e:
                print(f"Error deleting directory {dir_path}: {e}")
        else:
            print(f"Directory {dir_path} not found or is output directory, skipping.")

# Main execution
if __name__ == "__main__":
    # Download and extract datasets
    download_and_extract()
    
    # Resample audio for each split (uncommented to ensure resampling)
    for split, audio_dir in [("train", train_audio_dir), ("val", val_audio_dir), ("test", test_audio_dir)]:
        if os.path.exists(audio_dir):
            resample_audio(audio_dir, output_audio_dir)
        else:
            print(f"Audio directory {audio_dir} not found, skipping resampling.")
    
    # Process files and create manifests
    train_entries = process_files(train_audio_dir, train_text_file, output_audio_dir)
    write_manifest(train_entries, train_manifest)
    val_entries = process_files(val_audio_dir, val_text_file, output_audio_dir)
    write_manifest(val_entries, val_manifest)
    test_entries = process_files(test_audio_dir, test_text_file, output_audio_dir)
    write_manifest(test_entries, test_manifest)
    
    clean_remaining_file()
    # Write manifest files
    print(f"Total entries processed - Train: {len(train_entries)}, Validation: {len(val_entries)}, Test: {len(test_entries)}")