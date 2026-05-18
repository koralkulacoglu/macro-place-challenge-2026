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
# Uses heredoc to avoid bash expanding cmake variable references like TARGET_NAME.
python3 - "$SRC" << 'PYEOF'
import re, sys

def disable_pin_pos_segment(cmake_path):
    content = open(cmake_path).read()
    # Wrap only the add_pytorch_extension block for cuda_segment in if(FALSE);
    # pin_pos_cuda (no CUB) stays enabled for GPU global placement.
    # The cmake file uses ${TARGET_NAME}_cuda_segment variables (not literal strings).
    pattern = r'(add_pytorch_extension\(\$\{TARGET_NAME\}_cuda_segment[^)]*\))'
    replacement = 'if(FALSE)  # koral: disabled (CUDA 12.4 CUB incompatible)\n\\1\nendif()'
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if new_content == content:
        print(f'  WARNING: pin_pos_cuda_segment pattern not found in {cmake_path}')
        return
    # Remove cuda_segment from install(TARGETS ...) so only pin_pos_cuda is installed
    new_content = re.sub(r'[ \t]*\$\{TARGET_NAME\}_cuda_segment\n?', '', new_content)
    open(cmake_path, 'w').write(new_content)
    print(f'  Disabled pin_pos_cuda_segment in {cmake_path} (pin_pos_cuda kept intact)')

def disable_block(cmake_path, target_name):
    content = open(cmake_path).read()
    blocks = re.split(r'(?=if\s*\((?:TORCH_ENABLE_CUDA|torch_enable_cuda)\))', content)
    result = []
    for block in blocks:
        if re.match(r'if\s*\((?:TORCH_ENABLE_CUDA|torch_enable_cuda)\)', block) and target_name in block:
            block = re.sub(r'^if\s*\((?:TORCH_ENABLE_CUDA|torch_enable_cuda)\)',
                           'if(FALSE) # koral: disabled (CUDA 12.4 CUB incompatible)', block, count=1)
            print(f'  Block-disabled {target_name} in {cmake_path}')
        result.append(block)
    open(cmake_path, 'w').write(''.join(result))

src = sys.argv[1]
disable_pin_pos_segment(src + '/dreamplace/ops/pin_pos/CMakeLists.txt')
disable_block(src + '/dreamplace/ops/k_reorder/CMakeLists.txt', 'k_reorder')
disable_block(src + '/dreamplace/ops/global_swap/CMakeLists.txt', 'global_swap')
disable_block(src + '/dreamplace/ops/independent_set_matching/CMakeLists.txt', 'independent_set_matching')
PYEOF

# Patch 3: NumPy 2.0 compatibility
find "$SRC/dreamplace" -name "*.py" -exec sed -i 's/np\.string_/np.bytes_/g' {} \;
echo "  Fixed np.string_ in all DREAMPlace Python files"

# Patch 4: Soft-import fallbacks for disabled CUDA modules
python3 - "$SRC" << 'PYEOF'
import re, os, sys

src = sys.argv[1]
disabled_mods = [
    'pin_pos_cuda_segment',
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
        p1 = r'([ \t]*)(import\s+dreamplace\.ops\.[^\n]*\b' + re.escape(mod) + r'\b[^\n]*\n)'
        r1 = r'\1try:\n\1    \2\1except (ImportError, Exception):\n\1    ' + mod + ' = None\n'
        new = re.sub(p1, r1, content)
        p2 = r'([ \t]*)(from\s+\.\s+import\s+' + re.escape(mod) + r'\b[^\n]*\n)'
        r2 = r'\1try:\n\1    \2\1except (ImportError, Exception):\n\1    ' + mod + ' = None\n'
        new = re.sub(p2, r2, new)
        p3 = r'([ \t]*)(from\s+\.' + re.escape(mod) + r'\s+import\s+[^\n]*\n)'
        r3 = r'\1try:\n\1    \2\1except (ImportError, Exception):\n\1    pass\n'
        new = re.sub(p3, r3, new)
        if new != content:
            content = new
            modified = True
            print(f'  Patched import of {mod} in {op_path}')
    if modified:
        open(op_path, 'w').write(content)
PYEOF

echo "All patches applied successfully."
