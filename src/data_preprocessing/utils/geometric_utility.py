def circle_through_three_points(points):
     # Extract coordinates for clarity
    x1, y1 = points[0, :]
    x2, y2 = points[1, :]
    x3, y3 = points[2, :]

    # Calculate midpoints of lines joining (p1, p2) and (p2, p3)
    midx1 = (x1 + x2) / 2
    midy1 = (y1 + y2) / 2
    midx2 = (x2 + x3) / 2
    midy2 = (y2 + y3) / 2

    # Slopes of lines
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x3 - x2
    dy2 = y3 - y2

    # Handling perpendicular bisectors
    if dy1 == 0:  # Handle vertical line case for first line segment
        cx = midx1
        cy = -dx2 / dy2 * (cx - midx2) + midy2
    elif dy2 == 0:  # Handle vertical line case for second line segment
        cx = midx2
        cy = -dx1 / dy1 * (cx - midx1) + midy1
    else:
        slope1 = -dx1 / dy1
        slope2 = -dx2 / dy2

        # Intersection of two perpendicular bisectors
        cx = (slope1 * midx1 - slope2 * midx2 + midy2 - midy1) / (slope1 - slope2)
        cy = slope1 * (cx - midx1) + midy1

    # Calculate the radius using PyTorch operations
    radius = torch.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)

    return torch.tensor([cx, cy]), radius

