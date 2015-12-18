"""Algorithm for cointinuous data"""
from sklearn import gaussian_process
import numpy as np


def gaussian_process_regression(ts, data):

    ts_gauss = []
    data_gauss = []

    for idx in range(len(ts)):
        gp = gaussian_process.GaussianProcess(theta0=1e-2,
                                              thetaL=1e-4,
                                              thetaU=1e-1,
                                              nugget=1e-6)
        X = np.atleast_2d(ts[idx]).T
        y = data[idx]
        gp.fit(X, y)
        x = np.atleast_2d(np.linspace(0., 1., 100)).T
        y_pred, sigma2pred = gp.predict(x, eval_MSE=True)
        ts_gauss.append(x)
        data_gauss.append(y_pred)

    return (ts_gauss, data_gauss)

def time_window(timestamp, data, window_width, overlap):
    min_time = min([min(ts) for ts in timestamp])
    max_time = max([max(ts) for ts in timestamp])

    items = []

    for idx in 

    
    pass

    
