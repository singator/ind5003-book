import matplotlib.pyplot as plt
import statsmodels.api as sm


def check_normality(pd_series):
    """Creates a two-panel figure (histogram + QQ-plot) for checking normality."""
    plt.figure(1, figsize=(8, 3))
    plt.subplot(121)
    pd_series.hist(grid=False)
    tmp = plt.subplot(122)
    sm.qqplot(pd_series, line="q", ax=tmp)
