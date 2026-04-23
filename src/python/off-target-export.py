from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import requests, PIL, io, torch
import openvino as ov

print("OpenVINO version:", ov.__version__)
print("Pytorch version:", torch.__version__)
print("Available devices:", ov.Core().available_devices)

# Load image and test PyTorch model
img = PIL.Image.open(io.BytesIO(requests.get("https://placecats.com/300/200").content))
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights) 
model.eval()
preprocess = weights.transforms()
batch = preprocess(img).unsqueeze(0)

# PyTorch baseline inference
print("\nTesting PyTorch model on CPU...")
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"PyTorch (CPU) Output: {category_name}: {100 * score:.1f}%")

# Convert to OpenVINO and compile for Lunar Lake NPU (offline)
print("\nConverting model to OpenVINO format...")
core = ov.Core()
ov_model = ov.convert_model(model, example_input=batch)
ov_model.reshape([1, 3, 224, 224])

# Verify OpenVINO inference on off-target system, ex Xeon (CPU)
print("\nCompiling and Testing OV model for CPU...")
ov_compiled_model = core.compile_model(ov_model, "CPU")
prediction = torch.tensor(ov_compiled_model(batch)[0]).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"OpenVINO (CPU) Output: {category_name}: {100 * score:.1f}%")

# Configure for Lunar Lake (platform 4000) offline compilation
config = {
    "NPU_PLATFORM": "4000",  # Target Lunar Lake platform
    "NPU_COMPILER_TYPE": "PLUGIN",  # Force Compiler-In-Plugin
}
path_to_blob = "./xeon_npu_cache/resnet50_src_xeon_target_LNL.blob"

# Compile and export blob
print("\nCompiling model for NPU and exporting blob...")
ov_compiled_model = core.compile_model(ov_model, "NPU", config)
buffer = ov_compiled_model.export_model()
with open(path_to_blob, "wb") as f:
    f.write(buffer.getvalue())

print(f"Blob exported to {path_to_blob}")
