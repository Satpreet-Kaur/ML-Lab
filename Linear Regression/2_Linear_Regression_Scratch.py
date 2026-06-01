# Linear Regression from Scratch
import numpy as np

# Dataset
x = np.array([1,2,3,4,5])
y = np.array([2,4,6,8,14])

xmean = np.mean(x)
ymean = np.mean(y)

y1 = 0; x2 = 0;
for i in range(len(x)):
  y1 += (y[i] - ymean)*(x[i]-xmean);  #numerator
  x2 += (x[i]-xmean)**2   #denominator

m = y1/x2   #slope

b = ymean - m*xmean   #intercept

print(f'slope: {m}')
print(f'intercept: {b}')

predy = m*x[3] + b  #predicted value of y for x[3]
print(f'Predicted y: {predy}, original y: {y[3]}')
print((predy-y[3])/y[3])  #Error ratio
