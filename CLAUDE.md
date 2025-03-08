# Letta Development Guide

## Commands
- Install dependencies: `poetry install --all-extras`
- Run server: `poetry run letta run`
- Format code: `poetry run black . -l 140 && poetry run isort .`
- Lint: `poetry run pre-commit run --all-files`
- Run all tests: `poetry run pytest -s tests`
- Run specific test: `poetry run pytest -s tests/path/to/test_file.py::test_function_name`
- Run tests with marker: `poetry run pytest -s -m "marker_name" tests/`
- Create DB migration: `poetry run alembic revision --autogenerate -m "Migration message"`
- Apply migrations: `poetry run alembic upgrade head`

## Code Style Guidelines
- **Imports**: Group by standard library, third-party, local modules; use absolute imports
- **Types**: Use typing annotations consistently; resolve circular imports with TYPE_CHECKING
- **Naming**: Classes=PascalCase, functions/variables=snake_case, constants=UPPER_SNAKE_CASE
- **Structure**: Use Pydantic models for data validation; include docstrings
- **Error Handling**: Custom exceptions inherit from LettaError; provide descriptive messages
- **Line Length**: 140 characters maximum
- **Formatting**: Black and isort for consistent style