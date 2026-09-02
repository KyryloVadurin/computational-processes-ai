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


def on_page_markdown(markdown, page, config, files):
    # 1. Автоматично додаємо markdown="1" до всіх HTML-тегів <details>, щоб Markdown/MathJax парсився всередині
    markdown = re.sub(r'<details(?![^>]*markdown="1")', r'<details markdown="1"', markdown)

    # 2. Обробка та виправлення синтаксису Mermaid
    markdown = auto_wrap_mermaid_text(markdown)

    # 3. Перетворення блоків ```math на $$ ... $$
    markdown = re.sub(r'```math\s*\n([\s\S]*?)\n```', r'\n$$\n\1\n$$\n', markdown)

    # 4. Автоматичне додавання порожнього рядка перед списками, якщо його немає
    markdown = re.sub(r'([^\n])\n([ \t]*[-*+]|\d+\.)\s+', r'\1\n\n\2 ', markdown)

    # 5. Заміна знаку '<' у формулах $$ на '\lt', щоб не конфліктувало з HTML
    def fix_math_tags(match):
        math_content = match.group(0)
        return re.sub(r'<(\s*[0-9a-zA-Z_\\])', r'\\lt \1', math_content)

    markdown = re.sub(r'\$\$[\s\S]*?\$\$', fix_math_tags, markdown)

    return markdown
