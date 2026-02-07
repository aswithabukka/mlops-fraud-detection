# Contributing to MLOps Fraud Detection Pipeline

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## 🎯 Ways to Contribute

- 🐛 **Bug Reports**: Found a bug? Open an issue with details
- ✨ **Feature Requests**: Have an idea? Suggest it in issues
- 📝 **Documentation**: Improve docs, add examples
- 💻 **Code**: Fix bugs, add features, improve performance
- 🧪 **Testing**: Add tests, improve coverage
- 📊 **Examples**: Add notebooks, demo scripts

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/mlops-fraud-detection.git
cd mlops-fraud-detection
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or a bugfix branch
git checkout -b fix/bug-description
```

## 📋 Development Workflow

### Code Quality Standards

This project uses several tools to maintain code quality:

- **Black**: Code formatting (line length 100)
- **isort**: Import sorting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Bandit**: Security checks

### Before Committing

```bash
# Format code
black src/ tests/
isort src/ tests/

# Run linters
flake8 src/ tests/
mypy src/

# Run tests
pytest tests/ -v --cov=src

# Or use make commands
make format
make lint
make test
```

### Pre-commit Hooks

Pre-commit hooks will run automatically on `git commit`. They check:
- Code formatting (Black, isort)
- Linting (Flake8)
- Security (Bandit)
- YAML/JSON validation
- Trailing whitespace
- Large files
- Secrets detection

If hooks fail, fix the issues and commit again.

## 🧪 Testing Guidelines

### Writing Tests

- **Location**: `tests/unit/` for unit tests, `tests/integration/` for integration tests
- **Naming**: Test files should be named `test_*.py`
- **Coverage**: Aim for >60% coverage for new code
- **Fixtures**: Use pytest fixtures in `tests/conftest.py`

### Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Specific test
pytest tests/unit/test_generator.py::TestFraudDataGenerator::test_fraud_rate -v
```

## 📝 Commit Message Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```bash
feat(api): add batch prediction endpoint

- Add /predict/batch endpoint for processing multiple transactions
- Add request validation with Pydantic
- Update API documentation

Closes #42

---

fix(monitoring): correct drift threshold calculation

The drift threshold was using wrong comparison operator.
Changed from > to >= to match documentation.

---

docs(readme): update installation instructions

Added troubleshooting section for common installation issues.
```

## 🔄 Pull Request Process

### 1. Update Your Branch

```bash
# Fetch latest changes from main
git fetch upstream
git rebase upstream/main

# Or if you haven't added upstream
git remote add upstream https://github.com/aswithabukka/mlops-fraud-detection.git
git fetch upstream
git rebase upstream/main
```

### 2. Push Changes

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

- Go to GitHub and create a pull request
- Fill in the PR template
- Link related issues
- Wait for CI checks to pass
- Request review

### PR Title Format

```
feat: Add support for real-time streaming predictions
fix: Resolve memory leak in data preprocessing
docs: Update deployment guide for AWS
```

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## How Has This Been Tested?
Describe the tests you ran

## Checklist
- [ ] My code follows the style guidelines (Black, Flake8)
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Screenshots (if applicable)
Add screenshots to help explain your changes
```

## 🐛 Reporting Bugs

### Before Submitting a Bug Report

- Check existing issues to avoid duplicates
- Try the latest version of the code
- Collect relevant information (OS, Python version, error messages)

### Bug Report Template

```markdown
## Bug Description
A clear and concise description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:
1. Run command '...'
2. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.5]
- Docker version: [if applicable]

## Error Logs
```
Paste error logs here
```

## Additional Context
Any other context about the problem.
```

## ✨ Feature Requests

### Feature Request Template

```markdown
## Feature Description
Clear description of the feature.

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Solution
How you envision this feature working.

## Alternatives Considered
Other approaches you've thought about.

## Additional Context
Any other context, screenshots, or examples.
```

## 📚 Documentation

### Where to Add Documentation

- **Code**: Add docstrings to all functions and classes
- **README**: Update README.md for user-facing changes
- **GUIDE.md**: Add technical details for complex features
- **DEPLOYMENT_GUIDE.md**: Update for infrastructure changes
- **API Docs**: FastAPI auto-generates docs from docstrings

### Documentation Style

```python
def calculate_fraud_score(transaction: Dict[str, Any]) -> float:
    """
    Calculate fraud probability score for a transaction.

    This function uses the trained model to predict fraud probability
    based on transaction features.

    Args:
        transaction: Dictionary containing transaction features
            - amount (float): Transaction amount in USD
            - merchant_category (str): Merchant category
            - hour_of_day (int): Hour of transaction (0-23)

    Returns:
        float: Fraud probability score (0.0 to 1.0)

    Raises:
        ValueError: If required features are missing

    Example:
        >>> transaction = {"amount": 100.0, "merchant_category": "retail", ...}
        >>> score = calculate_fraud_score(transaction)
        >>> print(score)
        0.23
    """
    # Implementation
```

## 🏆 Recognition

Contributors will be:
- Listed in the project README
- Mentioned in release notes
- Recognized in the CONTRIBUTORS.md file

## 📞 Getting Help

- 💬 **Discussions**: Use GitHub Discussions for questions
- 🐛 **Issues**: Use GitHub Issues for bugs and features
- 📧 **Email**: Contact maintainers for sensitive issues

## 📄 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all.

### Our Standards

**Positive behavior:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

**Unacceptable behavior:**
- Harassment, trolling, or insulting comments
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Violations can be reported to the project maintainers. All complaints will be reviewed and investigated promptly and fairly.

## 🙏 Thank You!

Thank you for contributing to making this project better! Every contribution, no matter how small, is valuable.

---

**Questions?** Open an issue or start a discussion!
