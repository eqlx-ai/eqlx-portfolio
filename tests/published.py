"""Figures printed in public documents, used as test values.

Source: CreditMetrics - Technical Document (J.P. Morgan, 2 April 1997; Gupton, Finger, Bhatia).
Table numbers refer to that document.
"""

import numpy as np

RATINGS = ("AAA", "AA", "A", "BBB", "BB", "B", "CCC", "Default")

# Table 1.8 / 2.1: one-year transition matrix (%), Standard & Poor's CreditWeek, 15 April 1996.
TRANSITION = (
    np.array(
        [
            [90.81, 8.33, 0.68, 0.06, 0.12, 0.00, 0.00, 0.00],
            [0.70, 90.65, 7.79, 0.64, 0.06, 0.14, 0.02, 0.00],
            [0.09, 2.27, 91.05, 5.52, 0.74, 0.26, 0.01, 0.06],
            [0.02, 0.33, 5.95, 86.93, 5.30, 1.17, 0.12, 0.18],
            [0.03, 0.14, 0.67, 7.73, 80.53, 8.84, 1.00, 1.06],
            [0.00, 0.11, 0.24, 0.43, 6.48, 83.46, 4.07, 5.20],
            [0.22, 0.00, 0.22, 1.30, 2.38, 11.24, 64.86, 19.79],
        ]
    )
    / 100.0
)
AAA, AA, A, BBB, BB, B, CCC = range(7)

# Table 2.3: one-year forward zero curves by rating category (%), years 1 to 4.
FORWARD_CURVES = (
    np.array(
        [
            [3.60, 4.17, 4.73, 5.12],
            [3.65, 4.22, 4.78, 5.17],
            [3.72, 4.32, 4.93, 5.32],
            [4.10, 4.67, 5.25, 5.63],
            [5.55, 6.02, 6.78, 7.27],
            [6.05, 7.02, 8.03, 8.52],
            [15.05, 15.02, 14.03, 13.52],
        ]
    )
    / 100.0
)
RECOVERY_SENIOR_UNSECURED = 0.5113  # Table 2.2, mean
RECOVERY_SENIOR_UNSECURED_SD = 25.45  # Table 2.2, standard deviation, per 100 of face value

# Table 1.2: year-end values of the 5-year 6% BBB bond; Table 1.4: of the 3-year 5% single-A bond.
BBB_BOND_VALUES = np.array([109.37, 109.19, 108.66, 107.55, 102.02, 98.10, 83.64, 51.13])
A_BOND_VALUES = np.array([106.59, 106.49, 106.30, 105.64, 103.15, 101.39, 88.71, 51.13])

# Table 3.1: joint migration probabilities (%) of the BBB (rows) and A (columns) obligors, zero correlation.
JOINT_ZERO = np.array(
    [
        [0.00, 0.00, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.01, 0.30, 0.02, 0.00, 0.00, 0.00, 0.00],
        [0.01, 0.14, 5.42, 0.33, 0.04, 0.02, 0.00, 0.00],
        [0.08, 1.98, 79.15, 4.80, 0.64, 0.23, 0.01, 0.05],
        [0.00, 0.12, 4.83, 0.29, 0.04, 0.01, 0.00, 0.00],
        [0.00, 0.03, 1.06, 0.06, 0.01, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.11, 0.01, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.16, 0.01, 0.00, 0.00, 0.00, 0.00],
    ]
)
# Table 3.2 (= Table 1.7): the same with 0.30 asset correlation.
JOINT_030 = np.array(
    [
        [0.00, 0.00, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.04, 0.29, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.02, 0.39, 5.44, 0.08, 0.01, 0.00, 0.00, 0.00],
        [0.07, 1.81, 79.69, 4.55, 0.57, 0.19, 0.01, 0.04],
        [0.00, 0.02, 4.47, 0.64, 0.11, 0.04, 0.00, 0.01],
        [0.00, 0.00, 0.92, 0.18, 0.04, 0.02, 0.00, 0.00],
        [0.00, 0.00, 0.09, 0.02, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.13, 0.04, 0.01, 0.00, 0.00, 0.00],
    ]
)
# Table 8.7: BB (rows) and A (columns) obligors with 0.20 asset correlation (%). The document
# computed it from thresholds rounded to two decimals, so its margins are off by up to 0.04.
JOINT_BB_A_020 = np.array(
    [
        [0.00, 0.00, 0.03, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.01, 0.13, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.04, 0.61, 0.01, 0.00, 0.00, 0.00, 0.00],
        [0.02, 0.35, 7.10, 0.20, 0.02, 0.01, 0.00, 0.00],
        [0.07, 1.79, 73.65, 4.24, 0.56, 0.18, 0.01, 0.04],
        [0.00, 0.08, 7.80, 0.79, 0.13, 0.05, 0.00, 0.01],
        [0.00, 0.01, 0.85, 0.11, 0.02, 0.01, 0.00, 0.00],
        [0.00, 0.01, 0.90, 0.13, 0.02, 0.01, 0.00, 0.00],
    ]
)

# Table 8.5 (BB obligor) and Table 8.6 (A obligor): asset return thresholds in standard deviations.
THRESHOLDS_BB = np.array([3.43, 2.93, 2.39, 1.37, -1.23, -2.04, -2.30])
THRESHOLDS_A = np.array([3.12, 1.98, -1.51, -2.30, -2.72, -3.19, -3.24])
# Table 10.2: thresholds of the BBB, A and CCC firms of the simulation example.
# The BBB column prints 2.78 for the second threshold; the probabilities of Table 10.1 give 2.70.
THRESHOLDS_BBB = np.array([3.54, 2.78, 1.53, -1.49, -2.18, -2.75, -2.91])
THRESHOLDS_CCC = np.array([2.86, 2.86, 2.63, 2.11, 1.74, 1.02, -0.85])

# Table 10.4: ten scenarios of standardized asset returns for the BBB, A and CCC firms;
# Table 10.5: the ratings they map to.
SCENARIO_RETURNS = np.array(
    [
        [-0.7769, -0.8750, -0.6874],
        [-2.1060, -2.0646, 0.2996],
        [-0.9276, 0.0606, 2.7068],
        [0.6454, -0.1532, -1.1510],
        [0.4690, -0.5639, 0.2832],
        [-0.1252, -0.5570, -1.9479],
        [0.6994, 1.5191, -1.6503],
        [1.1778, -0.6342, -1.7759],
        [1.8480, 2.1202, 1.1631],
        [0.0249, -0.4642, 0.3533],
    ]
)
SCENARIO_RATINGS = (
    ("BBB", "A", "CCC"),
    ("BB", "BBB", "CCC"),
    ("BBB", "A", "A"),
    ("BBB", "A", "Default"),
    ("BBB", "A", "CCC"),
    ("BBB", "A", "Default"),
    ("BBB", "A", "Default"),
    ("BBB", "A", "Default"),
    ("A", "AA", "B"),
    ("BBB", "A", "CCC"),
)

# Table 9.2: values ($mm) of the three bonds of chapter 9 in each year-end rating.
THREE_ASSET_VALUES = np.array(
    [
        [4.375, 4.368, 4.346, 4.302, 4.081, 3.924, 3.346, 2.125],
        [2.132, 2.130, 2.126, 2.113, 2.063, 2.028, 1.774, 1.023],
        [1.162, 1.161, 1.161, 1.157, 1.142, 1.137, 1.056, 0.551],
    ]
)
