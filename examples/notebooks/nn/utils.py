
def scale_up(z, x_min, x_max):
    """
    Scales up z \in [-1,1] to x \in [x_min,x_max]
    where z = (2 * (x - x_min) / (x_max - x_min)) - 1
    """

    return x_min + (z + 1) * (x_max - x_min) / 2


def scale_down(x, x_min, x_max):
    """
    Scales down x \in [x_min,x_max] to z \in [-1,1]
    where z = f(x) = (2 * (x - x_min) / (x_max - x_min)) - 1
    """

    return (2 * (x - x_min) / (x_max - x_min)) - 1