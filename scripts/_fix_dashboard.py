"""Fix deprecated Streamlit API usage across all dashboard pages."""
import pathlib

dashboard = pathlib.Path("src/dashboard")
for path in dashboard.rglob("*.py"):
    text = path.read_text(encoding="utf-8")
    original = text
    text = text.replace("width='stretch'", "use_container_width=True")
    text = text.replace('width="stretch"', "use_container_width=True")
    if text != original:
        path.write_text(text, encoding="utf-8")
        n = original.count("width='stretch'") + original.count('width="stretch"')
        print(f"  Fixed {n:2d} occurrences in {path.name}")

print("Done")
