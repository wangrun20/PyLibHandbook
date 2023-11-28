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


def t_cdf(x, n):
    """计算P(X<x), 其中X服从自由度为n的t分布"""
    p = stats.t.cdf(x, df=n)
    return p


def t_ppf(p, n):
    """计算x, 使得P(X<x)=p, 其中X服从自由度为n的t分布"""
    x = stats.t.ppf(p, df=n)
    return x


def F_cdf(x, m, n):
    """计算P(X<x), 其中X服从自由度为(m, n)的F分布"""
    p = stats.f.cdf(x, m, n)
    return p


def F_ppf(p, m, n):
    """计算x, 使得P(X<x)=p, 其中X服从自由度为(m, n)的F分布"""
    x = stats.f.ppf(p, m, n)
    return x


def comb(n, m):
    """计算组合数C_n^m, 即从n个不同元素中取m个元素并成一组的取法总数"""
    c = math.comb(n, m)
    return c


class HyperGeometric(object):
    def __init__(self, n, M, N):
        """
        超几何分布, N个球中M个红色, 从中不放回地无序地抽n个, 看抽出了几个红球
        """
        self.n, self.M, self.N = n, M, N

    def pmf(self, k):
        """
        计算抽出了k个红球的概率
        """
        p = comb(self.M, k) * comb(self.N - self.M, self.n - k) / comb(self.N, self.n)
        return p
