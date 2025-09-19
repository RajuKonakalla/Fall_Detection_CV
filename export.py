from ultralytics import YOLO
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Export a YOLO model to OpenVINO format.")

# Add the arguments
parser.add_argument(
    '--model',
    type=str,
    default='yolov8n.pt',
    help='Path to the .pt model file to export.'
)

# Parse the arguments
args = parser.parse_args()

# Load the YOLO model
print(f"Loading model: {args.model}")
model = YOLO(args.model)

# Export the model to OpenVINO format
print("Exporting model to OpenVINO format...")
model.export(format='openvino')

print("\nExport complete!")
print(f"The OpenVINO model has been saved in a directory named '{args.model.split('.')[0]}_openvino_model'")
