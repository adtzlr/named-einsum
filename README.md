# Named Einsum
> NumPy's Einsum, but with named subscripts.

# Usage
```python
import numpy as np
from neinsum import named_einsum

x = np.eye(3)
y = named_einsum("A_ij,B_kl")(A=x, B=x)

# this is equal to
z = np.einsum("ij,kl", x, x)
```