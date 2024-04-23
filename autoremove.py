import os
import glob
import shutil

def remove_files(pattern):
    files = glob.glob(pattern)
    for f in files:
        try:
            os.remove(f)
            print(f"Removed: {f}")
        except OSError as e:
            print(f"Error: {f} : {e.strerror}")

# Remove all .log files
remove_files('*.log')

# Remove all .solver files
remove_files('*.solver')

# Remove all .png files
remove_files('*.png')

# Remove all .out files
remove_files('*.out')

# Remove all .ilp files
remove_files('*.ilp')

# Remove cache
shutil.rmtree('__pycache__')