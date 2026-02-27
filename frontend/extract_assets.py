#!/usr/bin/env python3
"""Извлечь CSS из index.html в styles.css. Опционально удалить инлайн <style> из index.html.
Запуск из корня: python3 frontend/extract_assets.py
С выносом стилей (удалить блок <style>...</style>): python3 frontend/extract_assets.py --no-inline
"""
import argparse
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == "frontend":
    base = script_dir
else:
    base = os.path.join(script_dir, "frontend")
path_html = os.path.join(base, "index.html")
path_css = os.path.join(base, "styles.css")

parser = argparse.ArgumentParser(description="Извлечь CSS в styles.css, опционально убрать инлайн <style> из index.html")
parser.add_argument("--no-inline", action="store_true", help="Удалить блок <style>...</style> из index.html (оставить только <link href=\"styles.css\">)")
args = parser.parse_args()

with open(path_html, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Границы блока <style>...</style>
style_line_start = style_line_end = None
for i, line in enumerate(lines):
    if "<style>" in line:
        style_line_start = i
    if "</style>" in line:
        style_line_end = i
        break
if style_line_start is None or style_line_end is None or style_line_end < style_line_start:
    style_line_end = min(style_line_start + 800, len(lines) - 1) if style_line_start is not None else 0
    style_line_start = style_line_start if style_line_start is not None else 0

# Извлечь CSS (содержимое между тегами)
content_start = style_line_start + 1
content_end = min(style_line_end, len(lines))
css_lines = [
    line[4:] if len(line) > 4 and line.startswith("    ") else line
    for line in lines[content_start:content_end]
]
with open(path_css, "w", encoding="utf-8") as out:
    out.write("".join(css_lines))
print("OK", path_css)

if getattr(args, "no_inline", False) and style_line_start is not None and style_line_end is not None and style_line_end >= style_line_start:
    new_lines = lines[: style_line_start] + lines[style_line_end + 1 :]
    with open(path_html, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("OK index.html: блок <style>...</style> удалён, остаётся только <link href=\"styles.css\">")
