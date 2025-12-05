import torch

# ✅ Use raw string for Windows path or double backslashes
MODEL_PATH = r"C:\Data\CORE\intrusion_detection_app\models\pytorch_model.pth"

# Step 1: Load using torch
model_data = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

# Step 2: (optional) Re-save safely
torch.save(model_data, r"C:\Data\CORE\intrusion_detection_app\models\pytorch_model_fixed.pth")

print("PyTorch model successfully re-saved as pytorch_model_fixed.pth")
