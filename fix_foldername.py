from pathlib import Path
import re
import os


def rename_brace_folders(base_dir="."):
    base_path = Path(base_dir)

    for folder in base_path.rglob("*"):
        if folder.is_dir():
            match = re.search(r"\{([^{}]*)\}", folder.name)
            if match:
                original_content = match.group(1)
                new_content = original_content.replace("_", ":")
                new_name = folder.name.replace(
                    f"{{{original_content}}}", f"{{{new_content}}}")
                new_path = folder.with_name(new_name)

                # 避免重名导致覆盖
                if not new_path.exists():
                    print(f"Renaming: {folder} -> {new_path}")
                    folder.rename(new_path)


rename_brace_folders(".")
