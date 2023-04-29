import pandas as pd


def make_data():
    Aa1 = [1, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    Aa2 = [1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    Bb1 = [0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    Bb2 = [0, 1, 0, 0, 0, 0, 1, 0, 0, 1]
    Cc1 = [0, 0, 1, 0, 0, 0, 0, 1, 0, 1]
    Cc2 = [0, 0, 1, 0, 0, 0, 0, 0, 1, 1]

    testdf = pd.DataFrame(
        columns=["A", "B", "C", "A1", "A2", "B1", "B2", "C1", "C2", "Alpha"]
    )

    testdf.loc[len(testdf.index)] = Aa1

    # adding Aa1 and Aa2
    for _ in range(125):
        testdf.loc[len(testdf.index)] = Aa1

    for _ in range(125):
        testdf.loc[len(testdf.index)] = Aa2

    # adding B
    for _ in range(125):
        testdf.loc[len(testdf.index)] = Bb1
    for _ in range(125):
        testdf.loc[len(testdf.index)] = Bb2

    # adding C
    for _ in range(450):
        testdf.loc[len(testdf.index)] = Cc1
    for _ in range(50):
        testdf.loc[len(testdf.index)] = Cc2

    struct = ((9,), (9,), (9,), (0,), (0,), (1,), (1,), (2,), (2,), ())

    return struct, testdf
