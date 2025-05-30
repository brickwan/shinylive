# coding=utf-8

import ast

def extract_imports(py_file):
    with open(py_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])

    return sorted(imports)

# 替换成你要分析的文件名
file_path = "shinylive20250530.py"

# 提取并写入 requirements.txt
packages = extract_imports(file_path)
with open("requirements.txt", "w") as f:
    for pkg in packages:
        f.write(pkg + "\n")

print("✅ 已根据 {} 生成 requirements.txt：".format(file_path))
print("\n".join(packages))
