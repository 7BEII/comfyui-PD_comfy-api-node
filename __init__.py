import importlib
import os


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def get_ext_dir(subpath=None):
    base_dir = os.path.dirname(__file__)
    if subpath is not None:
        base_dir = os.path.join(base_dir, subpath)
    return os.path.abspath(base_dir)


def safe_import_module(module_path, file_name):
    try:
        imported_module = importlib.import_module(module_path)

        if hasattr(imported_module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)
            print(f"[PD_NODE] Loaded module: {file_name}")

        if hasattr(imported_module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(imported_module.NODE_DISPLAY_NAME_MAPPINGS)

        return True
    except Exception as e:
        print(f"[PD_NODE] Failed to load module {file_name}: {e}")
        return False


py_dir = get_ext_dir("py")

if os.path.exists(py_dir):
    for file_name in os.listdir(py_dir):
        if not file_name.endswith(".py") or file_name.startswith("_"):
            continue

        module_name = os.path.splitext(file_name)[0]
        module_path = f"{__name__}.py.{module_name}"
        safe_import_module(module_path, file_name)
else:
    print(f"[PD_NODE] Missing py directory: {py_dir}")


WEB_DIRECTORY = "./js"

if NODE_CLASS_MAPPINGS:
    print("=" * 50)
    print("[PD_NODE] Comfyui-PD_comfy-api-node loaded")
    print("=" * 50)
    print(f"[PD_NODE] Total nodes: {len(NODE_CLASS_MAPPINGS)}")
    for node_name, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
        print(f"[PD_NODE] {node_name} -> {display_name}")
    print("=" * 50)
else:
    print("[PD_NODE] Warning: no nodes were loaded")


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
