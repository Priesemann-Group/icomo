SHELL := /bin/bash

.DEFAULT_GOAL := help

# RELEASE_APP=npx --yes \
# 	-p semantic-release \
# 	-p "@semantic-release/commit-analyzer" \
# 	-p "@semantic-release/release-notes-generator" \
# 	-p "@semantic-release/changelog" \
# 	-p "@semantic-release/exec" \
# 	-p "@semantic-release/github" \
# 	-p "@semantic-release/git" \
# 	-p "@google/semantic-release-replace-plugin" \
# 	semantic-release


PACKAGE_PATH="icomo"

SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = icomo
SOURCEDIR     = docs/
BUILDDIR      = docs/_build

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

.PHONY:help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY:clean
clean: ## remove build artifacts, compiled files, and cache
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {}  +
	find . -name '*~' -exec rm -f {} +
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

.PHONY:lint
lint: ## lint the project
	pre-commit run --all-files

.PHONY: test
test: ## run tests quickly with the default Python
	pytest -n auto


## run tests quickly with the latest packages, install uv before running this command
## uv doesn't have the same dependency resolution behavior as pip, so it may not lead
## to the same result as running `pip install -e .[dev]`, which is performed by
## CI of github.
.PHONY: test-latest
test-latest:
	uv venv .venv_test_setup
	uv pip install -e . --upgrade
	uv pip install -e .[dev]
	source .venv_test_setup/bin/activate
	pytest -n auto
	uv pip compile pyproject.toml --upgrade -o requirements.txt --quiet
	@echo "---------------------------------------"
	@echo "---Requirements.txt has been updated---"
	@echo "---------------------------------------"
	deactivate



.PHONY:docs-build
docs-build:
	sphinx-apidoc -o docs/_build ${PACKAGE_PATH}
	$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -P

.PHONY: docs-preview
docs-preview: docs-build
	cd docs/_build && python -m http.server

.PHONY:build
build:
	python -m build

# .PHONY:release-ci
# release-ci:
# 	$(RELEASE_APP) --ci
#
# .PHONY:release-dry
# release-dry:
# 	$(RELEASE_APP) --dry-run
