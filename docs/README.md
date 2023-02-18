# QubitApproximant

[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://pablovegan.github.io/QubitApproximant/)
[![pypi version](https://img.shields.io/pypi/v/mkdocstrings.svg)](https://pypi.org/project/mkdocstrings/)
[![conda version](https://img.shields.io/conda/vn/conda-forge/mkdocstrings)](https://anaconda.org/conda-forge/mkdocstrings)

A `python` package for approximating quantum circuits with a single qubit.

## Documentation and examples
Documentation created with `mkocs` can be found in https://pablovegan.github.io/QubitApproximant/.
## Installation

With `pip`:
```bash
pip install qubitapproximant
```

## Quick usage

Creating a quantum circuit with a cost function and optimizing the optimum parameters is very straightforward

```yaml
site_name: "My Library"

theme:
  name: "material"

plugins:
- search
- mkdocstrings
```

In one of your markdown files:

```markdown
# Reference

::: my_library.my_module.my_class
```

See the [Usage](https://mkdocstrings.github.io/usage) section of the docs for more examples!