import os
import struct
import sys

def convert_txt_to_bin(input_dir):
    """
    Finds all .txt files in the given directory that contain space-separated floats
    and creates a corresponding .bin file with raw float32 bytes for much faster C++ loading.
    """
    for file in os.listdir(input_dir):
        if file.endswith('.txt') and 'weight' in file or 'bias' in file:
            txt_path = os.path.join(input_dir, file)
            bin_path = txt_path[:-4] + '.bin'
            
            # Skip if .bin is already newer than .txt
            if os.path.exists(bin_path) and os.path.getmtime(bin_path) > os.path.getmtime(txt_path):
                continue
                
            print(f"Converting {file} to .bin...")
            
            with open(txt_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    continue
                values = [float(x) for x in content.split()]
                
            with open(bin_path, 'wb') as f:
                f.write(struct.pack(f'{len(values)}f', *values))

if __name__ == "__main__":
    weights_dir = "../weights"
    finetuned_dir = "../weights_finetuned"
    
    if os.path.exists(weights_dir):
        print(f"Checking {weights_dir} for text weights...")
        convert_txt_to_bin(weights_dir)
        
    if os.path.exists(finetuned_dir):
        print(f"Checking {finetuned_dir} for text weights...")
        convert_txt_to_bin(finetuned_dir)
        
    print("Done!")
