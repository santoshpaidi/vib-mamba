import os
import sys

def setup_environment():
    print("🚀 Setting up Waymo Open Dataset Protobufs...")
    
    # Download and install protoc
    protoc_version = "21.12"
    protoc_zip = f"protoc-{protoc_version}-linux-x86_64.zip"
    os.system(f"curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v{protoc_version}/{protoc_zip}")
    os.system(f"unzip -o {protoc_zip} -d /usr/local bin/protoc > /dev/null")
    os.system(f"unzip -o {protoc_zip} -d /usr/local 'include/*' > /dev/null")
    os.system(f"rm {protoc_zip}")
    
    # Clone and compile Waymo repo
    os.system("rm -rf waymo-open-dataset")
    os.system("git clone https://github.com/waymo-research/waymo-open-dataset.git")
    
    os.chdir("waymo-open-dataset")
    
    protos = [
        "label.proto", "dataset.proto", "protos/vector.proto", "protos/box.proto", 
        "protos/breakdown.proto", "protos/submission.proto", "protos/map.proto", 
        "protos/metrics.proto", "protos/camera_tokens.proto", "protos/keypoint.proto", 
        "protos/compressed_lidar.proto", "protos/end_to_end_driving_data.proto", 
        "protos/end_to_end_driving_submission.proto"
    ]
    
    for proto in protos:
        os.system(f"protoc --proto_path=src --python_out=src src/waymo_open_dataset/{proto}")
        
    print("✅ All proto files compiled!")

if __name__ == "__main__":
    setup_environment()