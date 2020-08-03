"""Face mark validation functions."""


def check_mark_location(image_height, image_width, marks):
    """Make sure all the marks are in the image. This function only check the
    x and y locations.

    Args:
        image_height: height of the image.
        image_width: width of the the image.
        marks: a numpy array of 3D marks.

    Returns:
        True if the marks are all in image, else False. 
    """
    if min(marks[:, 0]) < 0 or min(marks[:, 1]) < 0:
        return False
    if max(marks[:, 0]) > image_width or max(marks[:, 1]) > image_height:
        return False

    return True
