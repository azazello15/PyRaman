import numpy as np
import rampy as rp
import matplotlib.pyplot as plt
import matplotlib
import scipy  # For data manipulation
import random
import lmfit
from lmfit.models import GaussianModel
from numpy import argmax

matplotlib.use('TKAgg')

spectrum = np.genfromtxt("T20.txt", skip_header=10)  # Получение данных из файла
'''Первичная обработка массива данных'''
x = spectrum[:, 0]
y = spectrum[:, 1]

y_new = y + 1
y_norm = rp.normalise(y_new, method="minmax")

bir = np.array([(650, 750), (850, 900)])  # Выравнивание базовой линии
y_corr, y_base = rp.baseline(x, y_norm, bir, 'poly', polynomial_order=3)
lb = 720
hb = 950
x_fit = x[np.where((x > lb) & (x < hb))]
y_fit = y_corr[np.where((x > lb) & (x < hb))]
ese0 = np.sqrt(abs(y_fit[:, 0])) / abs(y_fit[:, 0])
y_fit[:, 0] = y_fit[:, 0] / np.amax(y_fit[:, 0]) * 10
sigma = abs(ese0 * y_fit[:, 0])

'''Функция для конечной обработки графика с использованием первичного местоположения пиков. 
Пики мы находим на глаз, но коррекция идет математически используя модель Гаусса'''


def residual(pars, x, data=None, eps=None):
    a1 = pars['a1'].value
    a2 = pars['a2'].value
    a3 = pars['a3'].value
    a4 = pars['a4'].value

    f1 = pars['f1'].value
    f2 = pars['f2'].value
    f3 = pars['f3'].value
    f4 = pars['f4'].value

    l1 = pars['l1'].value
    l2 = pars['l2'].value
    l3 = pars['l3'].value
    l4 = pars['l4'].value

    peak1 = rp.gaussian(x, a1, f1, l1)
    peak2 = rp.gaussian(x, a2, f2, l2)
    peak3 = rp.gaussian(x, a3, f3, l3)
    peak4 = rp.gaussian(x, a4, f4, l4)

    model = peak1 + peak2 + peak3 + peak4

    if data is None:
        return model, peak1, peak2, peak3, peak4
    if eps is None:
        return (model - data)
    return (model - data) / eps


'''тут как раз таки и идет коррекция пиков по базовой линии их нахождение и обработка'''
params = lmfit.Parameters()
#               (Name,  Value,  Vary,   Min,  Max,  Expr)
params.add_many(('a1', 10, True, 0, None, None),
                ('f1', 765, True, 755, 770, None),
                ('l1', 8, True, 5, 10, None),
                ('a2', 3.9, True, 0, None, None),
                ('f2', 775, True, 768, 785, None),
                ('l2', 4, True, 3, 6, None),
                ('a3', 4, True, 0, None, None),
                ('f3', 803, True, 790, 810, None),
                ('l3', 3.5, True, 2, 4, None),
                ('a4', 3.2, True, 0, None, None),
                ('f4', 815, True, 812, 840, None),
                ('l4', 5, True, 4, 6, None))

params['f1'].vary = False
params['f2'].vary = False
params['f3'].vary = False
params['f4'].vary = False

algo = 'nelder'

result = lmfit.minimize(residual, params, method=algo, args=(x_fit, y_fit[:, 0]))

params['f1'].vary = True
params['f2'].vary = True
params['f3'].vary = True
params['f4'].vary = True

result2 = lmfit.minimize(residual, params, method=algo, args=(x_fit, y_fit[:, 0]))

model = lmfit.fit_report(result2.params)
yout, peak1, peak2, peak3, peak4 = residual(result2.params, x_fit)

'''выявление пиков на спектре'''

plt.figure(figsize=(8, 6))
plt.plot(x_fit, y_fit, 'k-')
plt.plot(x_fit, yout, 'r-')
plt.plot(x_fit, peak1, 'b-')
plt.plot(x_fit, peak2, 'b-')
plt.plot(x_fit, peak3, 'b-')
plt.plot(x_fit, peak4, 'b-')

plt.xlim(lb, hb)
plt.ylim(-0.5, 10.5)
plt.xlabel("Raman shift, cm$^{-1}$", fontsize=14)
plt.ylabel("Normalized intensity, a. u.", fontsize=14)
plt.title("Fig. 3: Fit of the LiIO3 stretch vibrations\n in T20 with \nthe Nelder Mead algorithm ", fontsize=14,
          fontweight="bold")
plt.show()

'''положение максимумов спектра'''

v1 = x_fit[peak1.argmax()]
v2 = x_fit[peak2.argmax()]
v3 = x_fit[peak3.argmax()]
v4 = x_fit[peak4.argmax()]

'''интенсивность спектра'''

i1 = max(peak1)
i2 = max(peak2)
i3 = max(peak3)
i4 = max(peak4)

'''ширина спектральной линии'''



print(f'Положение максимума спектра:\n v1 = {v1}\n v2 = {v2}\n v3 = {v3}\n v4 = {v4}\n')
print(f'Интенсивность спектра:\n i1 = {i1}\n i2 = {i2}\n i3 = {i3}\n i4 = {i4}\n')
print(x_fit[peak1.argmin()])
