# Neinsum

A Named Einsum.

> *NumPy's Einsum, but with named subscripts.*

[![PyPI version shields.io](https://img.shields.io/pypi/v/neinsum.svg)](https://pypi.python.org/pypi/neinsum/) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://spdx.org/licenses/MIT.html) [![codecov](https://codecov.io/gh/adtzlr/named-einsum/graph/badge.svg?token=akiKR6sHEb)](https://codecov.io/gh/adtzlr/named-einsum) ![Codestyle black](https://img.shields.io/badge/code%20style-black-black)

# Installation
Neinsum is available on PyPI (and requires NumPy).

```
pip install neinsum
```

# Usage
With `neinsum`, it is possible to add names to the subscripts, i.e. instead of the indices-only `ij` in `np.einsum`, a named-subscript `A_ij` has to be provided. The variable names - like `A` (without indices) - are further used as keyword-arguments, see the example code-block. This is also supported for the output array.

```python
import numpy as np
from neinsum import named_einsum

x = np.eye(3)
y = np.arange(9).reshape(3, 3)

named_einsum("A_ij,B_kl")(A=x, B=y)

# this is equal to
np.einsum("ij,kl", x, y)
```
