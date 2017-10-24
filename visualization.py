import numpy as np
import lib.srtm as srtm
import matplotlib.pyplot as plt


def vis_stat(input_file):
    """ output statistics about the image data """
    # read the image
    a = srtm.read(input_file)

    # compute mean and standard deviation
    print(input_file + " : std: {0:.2f} mean: {1:.2f} meters".format(np.std(a), np.mean(a)))

    pass

def vis_load(input_file, output_file, size=(1024, 1024)):
    """read srtm file and write as jpeg"""
    # read the image
    a = srtm.read(input_file)
    # write it
    srtm.toimage(a, output_file, size)


def vis_mark(input_file, output_file, divisions, size=(1024, 1024)):
    """read  srtm file, mark subdivisions and write as jpeg"""
    # read the image
    a = srtm.read(input_file)
    # mark the image with subdivisions
    b = srtm.mark(a, divisions)
    # write it as small image to
    srtm.toimage(b, output_file, size)


def vis_subdivide(input_file, output_path, divisions, size=(96, 96)):
    # read the image
    a = srtm.read(input_file)
    # subdivide the image
    s = srtm.subdivide(a, divisions)

    i = 0
    for m in s:
        fname = output_path + "/a" + str(i) + ".jpg"
        srtm.toimage(m, fname, size)
        i += 1

    return s


def vis_datagen(input_file, output_path, divisions, size=(96, 96)):
    # read the image
    a = srtm.read(input_file)
    # subdivide the image
    s = srtm.subdivide(a, divisions)
    # pick one image
    m = s[0]
    # get 4 modified images
    r = srtm.datagen(m, 4, rotation_range=45, height_shift_range=0.1, width_shift_range=0.1)
    i = 1
    for t in r:
        a = srtm.tensor_to_array(t)
        fname = output_path + "/g" + str(i) + ".jpg"
        srtm.toimage(a, fname, size)
        i += 1


def vis_histogram(a_img):
    fig, ax = plt.subplots(1, 4, figsize=(6, 2))
    i = 0
    for aname in a_img:
        # read the file as a 1d array
        a = plt.imread(aname)
        # get one plane of monochrome image
        a = a[:, :, 0].flatten()
        h = ax[i].hist(a, rwidth=0.90, label=aname)
        ax[i].set_xlabel('pixel val')
        ax[i].set_ylabel('frequency')
        ax[i].plot()
        i += 1
    fig.set_tight_layout(True)
    fig.savefig("images/level1/histogram.svg")


if __name__ == "__main__":
    np.random.seed(42)
    base_files = [
        ["data/level1/N37W098.hgt", "images/level1/N37W098-b.jpg"],
        ["data/level1/N39W120.hgt", "images/level1/N39W120-b.jpg"],
        ["data/level2/N37W098.hgt", "images/level1/N37W098-b.jpg"],
        ["data/level2/N39W120.hgt", "images/level1/N39W120-b.jpg"],
    ]
    mark_files = [
        ["data/level1/N37W098.hgt", "images/level1/N37W098-m.jpg"],
        ["data/level1/N39W120.hgt", "images/level1/N39W120-m.jpg"]
    ]

   # for f in base_files:
   #     vis_stat(f[0])
#
   # for f in base_files:
   #     vis_load(f[0], f[1])
#
   # for f in mark_files:
   #     vis_mark(f[0], f[1], 5)
#
   # # create subdivided images
   # vis_subdivide("data/level1/N39W120.hgt", "images/level1", 5)
#
   # # generate mods of one image
   # vis_datagen("data/level1/N39W120.hgt", "images/level1", 5)

    # histograms
    a_img = [
        "images/level1/a0.jpg",
        "images/level1/a1.jpg",
        "images/level1/a2.jpg",
        "images/level1/a3.jpg",
    ]

    vis_histogram(a_img)
