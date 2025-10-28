# data/inspect_channel.py
import h5py

def inspect_h5(file_path='/home/diya/Projects/fluid-sr-thesis/data/raw/channel.h5'):
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Inspecting {file_path}")
            def print_structure(name, obj):
                indent = "  " * name.count('/')
                if isinstance(obj, h5py.Group):
                    print(f"{indent}Group: {name}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"{indent}Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
            f.visititems(print_structure)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_h5()