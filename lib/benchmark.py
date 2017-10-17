import lib.srtm as srtm


def run(X, y, groups):
    """solution model
    """
    print("benchmark")

    train_X, valid_X, test_X, train_y, valid_y, test_y = srtm.train_test_split(X, y, groups)

    print(train_X.shape, train_y.shape)
    print(valid_X.shape, valid_y.shape)
    print(test_X.shape, test_y.shape)
