#!/bin/bash
# Apply compatibility patches to a cloned DREAMPlace source tree.
# Usage: bash patch_dreamplace.sh /path/to/dreamplace_src
set -euo pipefail
SRC="${1:?Usage: $0 <dreamplace_src_dir>}"

# Patch 1: CUDA detection without a live GPU
sed -i \
  "s/print(int(torch.cuda.is_available()))/print(int(torch.version.cuda is not None))/" \
  "$SRC/cmake/TorchExtension.cmake"

# Patch 2: Disable CUDA ops incompatible with CUDA 12.4 CUB headers
sed -i 's/^if(TORCH_ENABLE_CUDA)/if(FALSE) # disabled/' \
  "$SRC/dreamplace/ops/pin_pos/CMakeLists.txt"
sed -i 's/^if (TORCH_ENABLE_CUDA)/if(FALSE) # disabled/' \
  "$SRC/dreamplace/ops/k_reorder/CMakeLists.txt"
sed -i 's/^if(TORCH_ENABLE_CUDA)/if(FALSE) # disabled/' \
  "$SRC/dreamplace/ops/global_swap/CMakeLists.txt"
sed -i 's/^if(TORCH_ENABLE_CUDA)/if(FALSE) # disabled/' \
  "$SRC/dreamplace/ops/independent_set_matching/CMakeLists.txt"

# Patch 3: NumPy 2.0 removed np.string_
sed -i 's/np\.string_/np.bytes_/g' "$SRC/dreamplace/PlaceDB.py"

# Patch 4: Soft-import disabled CUDA modules (try/except) via inline Python
DREAMPLACE_SRC="$SRC" python3 -c "
import re, os

src = os.environ['DREAMPLACE_SRC']
ops = {
    'pin_pos': 'pin_pos_cuda_segment',
    'global_swap': 'global_swap_cuda',
    'k_reorder': 'k_reorder_cuda',
    'independent_set_matching': 'independent_set_matching_cuda',
}
base = os.path.join(src, 'dreamplace', 'ops')
for op, mod in ops.items():
    path = os.path.join(base, op, op + '.py')
    if not os.path.exists(path):
        continue
    content = open(path).read()
    pattern = r'([ \t]*)(import dreamplace\.ops\.[^\n]*' + re.escape(mod) + r'[^\n]*\n)'
    repl = r'\1try:\n\1    \2\1except ImportError:\n\1    ' + mod + ' = None\n'
    content = re.sub(pattern, repl, content)
    open(path, 'w').write(content)
    print('Patched', path)
"

echo "All patches applied successfully."
