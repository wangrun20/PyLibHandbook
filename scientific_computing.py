import math
import scipy.stats as stats


def normal_cdf(x):
    """计算P(X<x), 其中X服从标准正态分布"""
    p = stats.norm.cdf(x)
    return p


def normal_ppf(p):
    """计算x, 使得P(X<x)=p, 其中X服从标准正态分布"""
    x = stats.norm.ppf(p)
    return x


def chis_cdf(x, n):
    """计算P(X<x), 其中X服从自由度为n的卡方分布"""
    p = stats.chi2.cdf(x, df=n)
    return p


def chis_ppf(p, n):
    """计算x, 使得P(X<x)=p, 其中X服从自由度为n的卡方分布"""
    x = stats.chi2.ppf(p, df=n)
    return x


def comb(n, m):
    """计算组合数C_n^m, 即从n个不同元素中取m个元素并成一组的取法总数"""
    c = math.comb(n, m)
    return c
