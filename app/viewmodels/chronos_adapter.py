import numpy as np

def aggregate_chronos_output(next_day, next_week, next_month):
    one_day = float(np.mean(next_day))
    one_week = list(np.mean(next_week, axis=0))
    one_month = list(np.mean(next_month, axis=0))

    return {
        "day": one_day,
        "week": one_week,
        "month": one_month
    }
