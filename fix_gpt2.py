import sys
import os

def replace_in_file(filepath, search, replace):
    with open(filepath, 'r') as f:
        content = f.read()
    if search not in content:
        print(f"Warning: search string not found in {filepath}")
        return False
    new_content = content.replace(search, replace)
    with open(filepath, 'w') as f:
        f.write(new_content)
    return True

# 1. Fix system.py initialization and multi-arch support
# (Done in previous steps, but let's make sure)
# 2. Fix pruned_attention.py forward signature and tuple handling
# 3. Fix statistics.py tuple handling
# 4. Fix evaluate.py max_samples
