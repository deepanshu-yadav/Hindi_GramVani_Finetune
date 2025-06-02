# First install the dependencies

Run this cell in the notebook.
```
# # Install dependencies
!pip install wget
!apt-get install sox libsndfile1 ffmpeg libsox-fmt-mp3
!pip install text-unidecode
!pip install matplotlib>=3.3.2

## Install NeMo
BRANCH = 'main'
!python -m pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[all]
```

After installation you need to restart the session for being able to use these libraries.

# Prepare the manifest file 

```
! cp Hindi_GramVani_Finetune/prepare_manifest.py .
! python prepare_manifest.py
```

# Tokenize the new Language
```
! cp Hindi_GramVani_Finetune/tokenize_language.py .
! python tokenize_language.py
```

# Download the parakeet v2 tdt  model and store it in some directory.
```
! cp Hindi_GramVani_Finetune/store_parakeet.py .
! python store_parakeet.py
```

# Fine Tune the model 

## Load the configuration
```
! cp Hindi_GramVani_Finetune/finetune.py .
! cp Hindi_GramVani_Finetune/hindi_config.yaml .
! cat hindi_config.yaml
```

## Finally finetune 
```
! python finetune.py
```