#!/bin/bash
# Apply compatibility patches to a cloned DREAMPlace source tree.
# Usage: bash patch_dreamplace.sh /path/to/dreamplace_src
set -eu
SRC="${1:?Usage: $0 <dreamplace_src_dir>}"

# Patch 1: CUDA detection without a live GPU device
sed -i \
  "s/print(int(torch.cuda.is_available()))/print(int(torch.version.cuda is not None))/" \
  "$SRC/cmake/TorchExtension.cmake"

# Patch 2: Disable CUB-based CUDA ops (CUDA 12.4 CUB API incompatible).
# pin_pos_cuda and pin_pos_cuda_segment share the same if(TORCH_ENABLE_CUDA) block
# in pin_pos/CMakeLists.txt, so disabling that block disables both. We accept CPU-only
# for pin_pos and compensate with the Python-level stub in placer.py (which redirects
# pin_pos_cuda imports to pin_pos_cpp at runtime).
# k_reorder, global_swap, independent_set_matching each have their own cmake file.
python3 -c "
import re, sys

def disable_block(cmake_path, target_name):
    '''Disable the TORCH_ENABLE_CUDA block containing target_name.'''
    content = open(cmake_path).read()
    blocks = re.split(r'(?=if\s*\((?:TORCH_ENABLE_CUDA|torch_enable_cuda)\))', content)
    result = []
    for block in blocks:
        if re.match(r'if\s*\((?:TORCH_ENABLE_CUDA|torch_enable_cuda)\)', block) and target_name in block:
            block = re.sub(r'^if\s*\((?:TORCH_ENABLE_CUDA|torch_enable_cuda)\)',
                           'if(FALSE) # koral: disabled (CUDA 12.4 CUB incompatible)', block, count=1)
            print(f'  Disabled {target_name} in {cmake_path}')
        result.append(block)
    open(cmake_path, 'w').write(''.join(result))

src = sys.argv[1]
# Disable the pin_pos CUDA block (includes both pin_pos_cuda and pin_pos_cuda_segment;
# pin_pos_cpp CPU fallback is used at runtime via Python-level stub in placer.py)
disable_block(src + '/dreamplace/ops/pin_pos/CMakeLists.txt', 'pin_pos_cuda_segment')
# Detailed-placement ops (CUB-based, unused with detailed_place_flag=0)
disable_block(src + '/dreamplace/ops/k_reorder/CMakeLists.txt', 'k_reorder')
disable_block(src + '/dreamplace/ops/global_swap/CMakeLists.txt', 'global_swap')
disable_block(src + '/dreamplace/ops/independent_set_matching/CMakeLists.txt', 'independent_set_matching')
" "$SRC"

# Patch 3: NumPy 2.0 compatibility — np.string_ removed, replaced by np.bytes_
# Fix ALL Python files in DREAMPlace (not just PlaceDB.py)
find "$SRC/dreamplace" -name "*.py" -exec sed -i 's/np\.string_/np.bytes_/g' {} \;
echo "  Fixed np.string_ in all DREAMPlace Python files"

# Patch 4: Soft-import disabled CUDA modules with proper fallbacks.
# Handles all import patterns (absolute, relative, from-import).
# Sets module to None on ImportError so callers can do 'if mod is not None: use CUDA'.
python3 -c "
import re, os, sys

src = sys.argv[1]

# Ops that were disabled at build time → need Python-level None fallback
disabled_mods = [
    'pin_pos_cuda_segment',  # disabled in pin_pos
    'k_reorder_cuda',
    'global_swap_cuda',
    'independent_set_matching_cuda',
]

base = os.path.join(src, 'dreamplace', 'ops')
for op_dir in os.listdir(base):
    op_path = os.path.join(base, op_dir, op_dir + '.py')
    if not os.path.exists(op_path):
        continue
    content = open(op_path).read()
    modified = False
    for mod in disabled_mods:
        if mod not in content:
            continue
        # Pattern 1: absolute import
        p1 = r'([ \t]*)(import\s+dreamplace\.ops\.[^\n]*\b' + re.escape(mod) + r'\b[^\n]*\n)'
        r1 = r'\1try:\n\1    \2\1except (ImportError, Exception):\n\1    ' + mod + ' = None\n'
        new = re.sub(p1, r1, content)
        # Pattern 2: relative from-import
        p2 = r'([ \t]*)(from\s+\.\s+import\s+' + re.escape(mod) + r'\b[^\n]*\n)'
        r2 = r'\1try:\n\1    \2\1except (ImportError, Exception):\n\1    ' + mod + ' = None\n'
        new = re.sub(p2, r2, new)
        # Pattern 3: from-dotmod import
        p3 = r'([ \t]*)(from\s+\.' + re.escape(mod) + r'\s+import\s+[^\n]*\n)'
        r3 = r'\1try:\n\1    \2\1except (ImportError, Exception):\n\1    pass\n'
        new = re.sub(p3, r3, new)
        if new != content:
            content = new
            modified = True
            print(f'  Patched import of {mod} in {op_path}')
    if modified:
        open(op_path, 'w').write(content)
" "$SRC"

echo "All patches applied successfully."
