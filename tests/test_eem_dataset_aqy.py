import numpy as np

from eempy.eem_processing.eem_dataset import EEMDataset


def test_aqy_returns_one_value_per_excitation_wavelength():
    eem_stack = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
        ]
    )
    ex_range = np.array([250.0, 260.0, 270.0])
    em_range = np.array([300.0, 310.0])
    abs_stack = np.array(
        [
            [1.0, 2.0, 5.0],
            [2.0, 4.0, 10.0],
        ]
    )
    dataset = EEMDataset(eem_stack=eem_stack, ex_range=ex_range, em_range=em_range, index=["s1", "s2"])

    aqy = dataset.aqy(abs_stack=abs_stack, ex_range_abs=ex_range)

    expected = np.array(
        [
            [3.0, 3.5, 2.2],
            [2.5, 2.25, 1.3],
        ]
    )
    assert list(aqy.columns) == [
        "AQY (ex = 250.0 nm)",
        "AQY (ex = 260.0 nm)",
        "AQY (ex = 270.0 nm)",
    ]
    np.testing.assert_allclose(aqy.to_numpy(), expected)


def test_aqy_target_ex_uses_closest_available_excitation():
    eem_stack = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
        ]
    )
    ex_range = np.array([250.0, 260.0, 270.0])
    em_range = np.array([300.0, 310.0])
    abs_stack = np.array(
        [
            [1.0, 2.0, 5.0],
            [2.0, 4.0, 10.0],
        ]
    )
    dataset = EEMDataset(eem_stack=eem_stack, ex_range=ex_range, em_range=em_range, index=["s1", "s2"])

    aqy_at_target = dataset.aqy(abs_stack=abs_stack, ex_range_abs=ex_range, target_ex=261)

    assert aqy_at_target.name == "AQY (ex = 260.0 nm)"
    np.testing.assert_allclose(aqy_at_target.to_numpy(), np.array([3.5, 2.25]))
