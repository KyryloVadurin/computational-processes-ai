import re

def auto_wrap_mermaid_text(markdown):
    """
    Виправляє лапки всередині блоків Mermaid та автопереносить довгі рядки.
    """
    def process_mermaid_block(match):
        code = match.group(1)

        # 1. Замінюємо вкладені подвійні лапки всередині назв вузлів на одинарні або лапки-ялинки
        # Приклад: ["Текст ("підтекст")"] -> ["Текст ('підтекст')"]
        def sanitize_internal_quotes(m):
            prefix = m.group(1)
            content = m.group(2)
            suffix = m.group(3)
            # Прибираємо внутрішні подвійні лапки
            cleaned_content = content.replace('"', "'")
            return f'{prefix}"{cleaned_content}"{suffix}'

        # Знаходимо всі конструкції ["..."], (...), {...}
        code = re.sub(r'(\[|\(|\{)\s*\"(.*?)\"\s*(\]|\)|\})', sanitize_internal_quotes, code)

        return f"```mermaid\n{code}\n```"

    return re.sub(r'```mermaid\s*\n([\s\S]*?)\n```', process_mermaid_block, markdown)


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
