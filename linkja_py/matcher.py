def dice_coefficient(bf1, bf2):
    intersection = sum([1 for item in bf1 if item in bf2])
    return 2 * intersection / (len(bf1) + len(bf2))
