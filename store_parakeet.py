import nemo.collections.asr as nemo_asr
import os
import wget
# Load the model
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

# Save the model to a local directory
model_dir = "parakeet-tdt-0.6b-v2"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "parakeet-tdt-0.6b-v2.nemo")
asr_model.save_to(model_path)