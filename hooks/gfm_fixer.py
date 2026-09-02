import re

def fix_details_and_lists(markdown):
    """
    Забезпечує порожні рядки всередині <details> та перед усіма списками.
    """
    # 1. Додаємо порожній рядок після <summary>...</summary>, якщо його немає
    markdown = re.sub(r'(</summary>)\s*\n(?!\n)', r'\1\n\n', markdown)

    # 2. Додаємо порожній рядок перед закриваючим </details>
    markdown = re.sub(r'([^\n])\n\s*(</details>)', r'\1\n\n\2', markdown)

    # 3. Гарантуємо порожній рядок перед списками (1., 2. або *, -), якщо перед ними йде звичайний текст
    markdown = re.sub(r'([^\n])\n([ \t]*(\d+\.|[-*+])\s+)', r'\1\n\n\2', markdown)

    # 4. Гарантуємо порожній рядок перед вкладеними зірочками після двокрапки (наприклад, "дистракторів:\n*")
    markdown = re.sub(r'(:\s*)\n([ \t]*[-*+]\s+)', r'\1\n\n\2', markdown)

    return markdown

def on_page_markdown(markdown, page, config, files):
    # Додаємо markdown="1" до <details>
    markdown = re.sub(r'<details(?![^>]*markdown="1")', r'<details markdown="1"', markdown)

    # Виправляємо переноси та списки
    markdown = fix_details_and_lists(markdown)

    # Обробка та виправлення синтаксису Mermaid
    markdown = auto_wrap_mermaid_text(markdown)

    # Перетворення ```math на $$
    markdown = re.sub(r'```math\s*\n([\s\S]*?)\n```', r'\n$$\n\1\n$$\n', markdown)

    # Заміна < у формулах на \lt
    def fix_math_tags(match):
        math_content = match.group(0)
        return re.sub(r'<(\s*[0-9a-zA-Z_\\])', r'\\lt \1', math_content)

    markdown = re.sub(r'\$\$[\s\S]*?\$\$', fix_math_tags, markdown)

    return markdown
