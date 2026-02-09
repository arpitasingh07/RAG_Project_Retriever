import json, sys

path = "Group61_ConvoAI.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove widget metadata if present
metadata = nb.get("metadata", {})
if "widgets" in metadata:
    print("Removing metadata.widgets...")
    metadata.pop("widgets")

if "extensions" in metadata:
    # remove widget info inside extensions as well
    if "jupyter" in metadata["extensions"]:
        ext = metadata["extensions"]["jupyter"]
        if "widgets" in ext:
            print("Removing extension widget metadata...")
            ext.pop("widgets")

# Write cleaned notebook
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Notebook fixed!")