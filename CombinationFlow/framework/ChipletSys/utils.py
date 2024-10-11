def is_overlap_with_rects(x_bl, y_bl, x_ur, y_ur, rects):
    """
        input: coordinates of bottom-left and upper-right
        Return: whether input rectangle overlap with rects
    """
    for (rx_bl, ry_bl, rx_ur, ry_ur) in rects:
        x_overlap = not (x_bl >= rx_ur or rx_bl >= x_ur)  # overlap in x-axis
        y_overlap = not (y_bl >= ry_ur or ry_bl >= y_ur)  # overlap in y-axis

        if x_overlap and y_overlap:
            return True
    return False

def is_rects_overlap(rects):
    """
        rects: list of coordinates, (x_bl, y_bl, x_ur, y_ur)
        True: at least two of the rects overlap
    """
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            x0_bl, y0_bl, x0_ur, y0_ur = rects[i]
            x1_bl, y1_bl, x1_ur, y1_ur = rects[j]

            x_overlap = not (x0_bl >= x1_ur or x1_bl >= x0_ur)  # overlap in x-axis
            y_overlap = not (y0_bl >= y1_ur or y1_bl >= y0_ur)  # overlap in y-axis

            if x_overlap and y_overlap:
                return True
    return False