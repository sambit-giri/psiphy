import psiphy


def test_package_imports():
    import psiphy.utils
    import psiphy.sbi
    import psiphy.mcmc
    import psiphy.forecasting
    import psiphy.plotting
    import psiphy.toy_models


def test_version_defined():
    assert psiphy.__version__ != ""


def test_toy_models_exposed():
    from psiphy.toy_models import NoisyLine, GaussianSignal
    assert NoisyLine is not None
    assert GaussianSignal is not None
