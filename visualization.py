import numpy as np
import lib.srtm as srtm


def vis_load(input_path, output_path, size=(1024, 1024)):
    """read srtm file and write as jpeg"""
    # read the image
    a = srtm.read(input_path)
    # write it
    srtm.toimage(a, output_path, size)


def vis_mark(input_path, output_path, divisions, size=(1024, 1024)):
    """read  srtm file, mark subdivisions and write as jpeg"""
    # read the image
    a = srtm.read(input_path)
    # mark the image with 8x8subdivisions
    b = srtm.mark(a, divisions)
    # write it as small image to
    srtm.toimage(b, output_path, size)


if __name__ == "__main__":
    base_files = [
        ["data/level2/N37W098.hgt", "images/level2/N37W098-b.jpg"],
        ["data/level2/N39W120.hgt", "images/level2/N39W120-b.jpg"]
    ]
    mark_files = [
        ["data/level2/N37W098.hgt", "images/level2/N37W098-m.jpg"],
        ["data/level2/N39W120.hgt", "images/level2/N39W120-m.jpg"]
    ]
    # process
    for f in base_files:
        vis_load(f[0], f[1])

    for f in mark_files:
        vis_mark(f[0], f[1], 8)
