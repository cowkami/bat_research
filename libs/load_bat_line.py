from pathlib import Path
import numpy as np
import pandas as pd



def load_xyz_from_exl(file_path):
    '''
    load bat flying line, xyz coodinations from excel sheet.

    parameters
    ----------
    file_path: Path

    returns
    -------
    xyzt: pd.Dataframe(include coodinations, and time[sec])
    '''
    df = pd.read_excel(file_path, sheet_name='3D')
    d = {
        'x': df['bat(X)'],
        'y': df['bat(Y)'],
        'z': df['bat(Z)'],
        't': df['time']
    }
    xyzt = pd.DataFrame(data=d)

    return xyzt


def polyfit(x, y):
    coef = np.polyfit(x[:-1], y[:-1], 10)
    f = np.poly1d(coef)

    return f(x)


def moving_average(signal, period):
    buff = [np.nan] * period

    for i in range(period,len(signal)):
        buff.append(signal[i-period:i].mean())

    return buff


def cal_derivative(xs, ys, dt, rank=1):
    '''
    get speed or acceleration array.
    
    parameters
    ----------
    xs: ndarray
    ys: ndarray
    dt: float(time[sec])
    rank: int
    
    returns
    -------
    xs: ndarray(derivative) 
    ys: ndarray(derivative) 
    dxy: ndarray(derivative) 
    '''
    dx = (xs[1:] - xs[:-1]) / dt
    dy = (ys[1:] - ys[:-1]) / dt
    velocity = np.sqrt(dx**2 + dy**2)

    if rank == 1:
        return velocity
    elif rank == 2:
        accel = (velocity[1:] - velocity[:-1]) / dt
        return accel
    else:
        print('set acculate rank!')

