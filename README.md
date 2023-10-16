# Neinsum (Named Einsum)
> *NumPy's Einsum, but with named subscripts.*

[![codecov](https://codecov.io/gh/adtzlr/named-einsum/graph/badge.svg?token=akiKR6sHEb)](https://codecov.io/gh/adtzlr/named-einsum)

# Installation
```
pip install neinsum
```

# Usage
```python
import numpy as np
from neinsum import named_einsum

x = np.eye(3)
y = named_einsum("A_ij,B_kl")(A=x, B=x)

# this is equal to
z = np.einsum("ij,kl", x, x)
```
