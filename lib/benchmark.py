import lib.srtm as srtm


def run(X, y, groups):
    """solution model
    """
    print("benchmark")

    srtm.train_test_split(X, y, groups)
