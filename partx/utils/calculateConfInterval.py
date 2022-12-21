from scipy import stats

def conf_interval(x, conf_at):
    """Calculate Confidence interval

    Args:
        x ([type]): [description]
        conf_at ([type]): [description]

    Returns:
        [type]: [description]
    """
    mean, std = x.mean(), x.std(ddof=1)
    conf_intveral = stats.norm.interval(conf_at, loc=mean, scale=std)
    
    return conf_intveral