from copy import copy
def remove_ellipsis(sequence):
    first_non_ellipsis = None
    for idx, el in enumerate(sequence):
        if el is not ...:
            first_non_ellipsis = el
            break
    if first_non_ellipsis is None:
        return sequence

    current_non_ellipsis = first_non_ellipsis
    for idx, el in enumerate(sequence):
        if el is not ...:
            current_non_ellipsis = el
        else:
            sequence[idx] = copy(current_non_ellipsis)
    return sequence