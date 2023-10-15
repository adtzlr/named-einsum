import numpy as np


def named_einsum(named_subscripts: str) -> callable:
    """
    NumPy's Einsum, but with named subscripts.

    Parameters
    ----------
    nsubscripts : str
        Specifies the named subscripts for summation as comma separated list of the name
        and the subscript labels, separated by an underscore. An implicit (classical
        Einstein summation) calculation is performed unless the explicit indicator ‘->’
        is included as well as subscript labels of the precise output form.

    Returns
    -------
    callable
        Evaluates the Einstein summation convention on the operands, which are given as
        keyword-arguments according to the names in the named subscripts.

    Examples
    --------
    >>> import numpy as np
    >>> from named_einsum import named_einsum

    >>> x = np.eye(3)
    >>> y = named_einsum("A_ij,B_kl")(A=x, B=x)

    This is equal to:

    >>> z = np.einsum("ij,kl", x, x)
    >>> np.allclose(y, z)
    True

    """

    def einsum(**kwargs):
        "A wrapper for NumPy's Einsum to work with named subscripts."

        # trim (remove all) whitespaces
        subs = "".join([s for s in named_subscripts.split(" ") if len(s) > 0])

        # create a list of all subscripts of named operands (both input and output)
        # e.g. ``[["A_ij", "B_kl"], ["C_ijkl"]]``
        list_of_named_subscripts = [s.split(",") for s in subs.split("->")]

        # m = 2 if explicit subscripts of the output are defined, m=1 otherwise
        m = len(list_of_named_subscripts)

        # init empty lists for keys and subscripts of the operands
        keys = [[], []][:m]
        subscripts = [[], []][:m]

        # loop over the lists of the input and the output subscripts
        for a, nsubscripts in enumerate(list_of_named_subscripts):
            # loop over the named subscripts
            for named_subscript in nsubscripts:
                # split a named subscript into a key and the indices and add them
                # to the respective lists of keys and subscripts
                key, subscript = named_subscript.split("_")
                keys[a].append(key)
                subscripts[a].append(subscript)

        # re-join subscripts without the names
        subscripts = "->".join([",".join(indices) for indices in subscripts])

        # extract the operands from the keyword arguments
        operands = [kwargs.pop(key) for key in keys[0]]

        # extract the output array
        if m > 1 and keys[1][0] in kwargs.keys():
            out = kwargs.pop(keys[1][0])
        elif "out" in kwargs.keys():
            out = kwargs.pop("out")
        else:
            out = None

        # evaluate and return the result of ``numpy.einsum``
        return np.einsum(subscripts, *operands, out=out, **kwargs)

    return einsum
