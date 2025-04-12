import nbformat
import re

from pathlib import Path

def extract_imports_from_notebook(notebook_path):
    nb = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell['source'] for cell in nb.cells if cell.cell_type == 'code']
    import_lines = []

    for cell in code_cells:
        lines = cell.split('\n')
        for line in lines:
            if re.match(r'^\s*(import|from)\s+', line):
                import_lines.append(line.strip())

    return import_lines

def extract_package_names(import_lines):
    packages = set()
    for line in import_lines:
        match = re.match(r'^\s*(?:from|import)\s+([\w\.]+)', line)
        if match:
            root_pkg = match.group(1).split('.')[0]
            packages.add(root_pkg)
    return sorted(packages)

def get_installed_versions(packages):
    installed = {}
    for pkg in packages:
        try:
            version = pkg_resources.get_distribution(pkg).version
            installed[pkg] = version
        except pkg_resources.DistributionNotFound:
            installed[pkg] = None  # not installed
    return installed

def write_requirements(installed, output_path="requirements.txt"):
    with open(output_path, 'w') as f:
        for pkg, version in installed.items():
            if version:
                f.write(f"{pkg}=={version}\n")
            else:
                f.write(f"# {pkg}  # not installed\n")
    print(f"✅ requirements.txt created at: {output_path}")

# 🔧 Replace this with your notebook path
notebook_file = "/project_ml_flow/notebooks/Price_Elasticity.ipynb"

# Run extraction
imports = extract_imports_from_notebook(notebook_file)
pkgs = extract_package_names(imports)
installed = get_installed_versions(pkgs)
write_requirements(installed)
