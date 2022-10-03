import gtsam


def Y(y: int) -> gtsam.symbol:
    """Convert an integer to a gtsam
    symbol object with a y

    Args:
        y (int): the key number

    Returns:
        gtsam.symbol: the int in symbol form, y:1 for example
    """

    return gtsam.symbol("y", y)


def Z(z: int) -> gtsam.symbol:
    """Convert an integer to a gtsam
    symbol object with a z

    Args:
        z (int): the key number

    Returns:
        gtsam.symbol: the int in symbol form, z:1 for example
    """

    return gtsam.symbol("z", z)
