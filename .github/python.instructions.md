---
applyTo: '**/*.py'
---

- **Dependency Management**: Currently, use `uv` to handle dependencies and python virtual environments.
- **Elegance and Readability:** Strive for elegant and Pythonic code that is easy to understand and maintain.
- **PEP 8 Compliance:** Adhere to PEP 8 guidelines for code style, with Ruff as the primary linter and formatter.
- **Comprehensive Type Annotations:** All functions, methods, and class members must have type annotations, using the most specific types possible.
- **Comments/Docstrings:** Only attach Google-style docstrings to function that is important and non-trivial to understand. Prefer single-line docstring for short / easy to understand functions for clarity.
- **Exception Handling**: Prefer letting error fail fast. Introduce try/except statements only if the code can handle the error properly. In this case, Use specific exception types, provide informative error messages, and handle exceptions gracefully. 
- **Logging:** Employ the `logging` module judiciously to log important events, warnings, and errors.
- **Data Validation:** Use Pydantic models for rigorous request and response data validation.
- **Typing**: Prefer native types, e.g., `list` over `List`, `dict` over `Dict`, `set` over `Set`. Try to provide as specific/concrete types as possible. Use `Any` unless absolutely necessary.