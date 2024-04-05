def random_diameter(mean=0.8, std_dev=0.03, min_diameter=0.7, max_diameter=0.9):
    """This function generates a random diameter value following a constrained gaussian distribution."""
    while True:
        diameter = random.gauss(mean, std_dev)
        if min_diameter <= diameter <= max_diameter:
            return diameter
        


def is_image_generally_bright(image, sample_points=5, threshold=30):
    """Check if the image is generally bright by sampling multiple points around the center."""
    width, height = image.size
    pixels_checked = 0
    bright_pixels = 0

    for dx in range(-sample_points // 2, sample_points // 2 + 1):
        for dy in range(-sample_points // 2, sample_points // 2 + 1):
            x = width // 2 + dx
            y = height // 2 + dy
            pixel = image.getpixel((x, y))
            brightness = sum(pixel) / 3
            pixels_checked += 1
            if brightness > threshold:
                bright_pixels += 1

    return bright_pixels / pixels_checked > 0.5  # more than half of the pixels are bright enough




def resample_and_convert_to_jpg(infname, outfname_tiff, outfname_jpg, minx, miny, maxx, maxy, dx, dy):
    """Resample the GeoTIFF image and convert it to JPEG, returning the image object."""
    box_str = f"{minx} {miny} {maxx} {maxy}"
    pixsp_str = f"{dx} {dy}"
    command = f'gdalwarp -overwrite -te {box_str} -tr {pixsp_str} -r max -ot byte -of Gtiff {infname} {outfname_tiff}'
    os.system(command)

    # Open the resampled TIFF, convert to RGB, and save as JPEG
    img = PILImage.open(outfname_tiff)
    img_rgb = img.convert('RGB')
    img_rgb.save(outfname_jpg, 'JPEG')

    return img_rgb