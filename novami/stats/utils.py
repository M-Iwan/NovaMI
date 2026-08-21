import numpy as np

def round_to_significant(value: float, n: int):
    """
    Round a value to a given number of significant digits.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return 'None'

    if value == 0:
        return 0
    else:
        return round(value, n - int(np.floor(np.log10(abs(value)))) - 1)