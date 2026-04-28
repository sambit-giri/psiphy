import psi


def test_package_imports():
    import psi.utils
    import psi.sbi
    import psi.mcmc
    import psi.forecasting
    import psi.plotting
    import psi.toy_models


def test_version_defined():
    assert psi.__version__ != ""


def test_toy_models_exposed():
    from psi.toy_models import NoisyLine, GaussianSignal
    assert NoisyLine is not None
    assert GaussianSignal is not None
