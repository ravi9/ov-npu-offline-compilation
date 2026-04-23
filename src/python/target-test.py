import openvino as ov
import torch
from torchvision.models import resnet50, ResNet50_Weights
import requests, PIL, io

print("OpenVINO version:", ov.__version__)
print("Pytorch version:", torch.__version__)
print("Available devices:", ov.Core().available_devices)

# Load the image for testing  
img = PIL.Image.open(io.BytesIO(requests.get("https://placecats.com/200/300").content))  
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()
batch = preprocess(img).unsqueeze(0)

# Configure for Lunar Lake (platform 4000)
print("\nLoading OV compiled model blob on NPU...")
core = ov.Core()
config = {
    "NPU_PLATFORM": "4000",  # Target Lunar Lake platform
    "NPU_COMPILER_TYPE": "PLUGIN",  # Force Compiler-In-Plugin
}  

path_to_blob = "./xeon_npu_cache/resnet50_src_xeon_target_LNL.blob"
with open(path_to_blob, "rb") as f:
    ov_compiled_model = core.import_model(f.read(), "NPU", config)

# Run inference on NPU
print("\nTesting OV model on NPU...")
prediction = torch.tensor(ov_compiled_model(batch)[0]).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"OpenVINO (NPU) Output: {category_name}: {100 * score:.1f}%")