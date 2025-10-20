# Contributing to Blood Discriminator

Thank you for considering contributing to the Blood Discriminator project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, etc.)
- Error messages or logs

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- Clear description of the enhancement
- Use cases and benefits
- Potential implementation approach

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code style guidelines
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Run tests** to ensure nothing breaks
6. **Submit a pull request** with a clear description

## Development Setup

1. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/Blood_discriminator.git
cd Blood_discriminator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install development dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

4. Run tests:
```bash
pytest tests/
```

## Code Style Guidelines

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for all functions/classes
- Keep functions focused and modular
- Add comments for complex logic

### Docstring Format

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    """
    pass
```

## Testing

- Write unit tests for new functionality
- Ensure tests pass before submitting PR
- Aim for good test coverage
- Test edge cases

Run tests:
```bash
pytest tests/ -v
```

## Documentation

- Update README.md if adding new features
- Update docstrings for modified functions
- Add examples for new functionality
- Update the Springer LNCS report if relevant

## Commit Messages

Use clear, descriptive commit messages:

```
Add feature: Brief description

Detailed description of what changed and why.
```

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
