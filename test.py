import numpy as np
# x = np.array([46, 53, 37, 42, 34, 29, 60, 44, 41, 48, 33, 40])
# y = np.array([12, 14, 11, 13, 10, 8, 17, 12, 10, 15, 9, 13])
x = np.array([71, 49, 80, 73, 93, 85, 58, 82, 64, 32, 87, 80])
y = np.array([83, 62, 76, 77, 89, 74, 48, 78, 76, 51, 73, 89])
n=len(x)

Sxy = np.sum(x*y) - (1/n)*(np.sum(x)*np.sum(y))
Sxx = np.sum(x**2) - (1/n)*(np.sum(x))**2
Syy = np.sum(y**2) - (1/n)*(np.sum(y))**2

print("Sxy =", Sxy, "Sxx = ", Sxx, "xbar = ", np.sum(x)/n, "ybar = ", np.sum(y)/n)
beta = Sxy/Sxx
r = Sxy/(Syy*Sxx)**(1/2)
t = 3.169
Gamma = (1-r**2)**(1/2)/(r*((n-2)**(1/2)))
print(f"r={r}, beta={beta}, Gamma={Gamma}")
print(beta*(1-t*Gamma), ", ", beta*(1+t*Gamma))
# x = np.array([50, 100, 250, 500, 1000])
# y = np.array([108, 53, 24, 9, 5])
# n=len(x)

# x = np.log10(x)
# y = np.log10(y)

# Sxy = np.sum(x*y) - (1/n)*(np.sum(x)*np.sum(y))
# Sxx = np.sum(x**2) - (1/n)*(np.sum(x))**2
# beta = Sxy/Sxx
# log_alpha = np.sum(x)/n - Sxy/Sxx*np.sum(y)/n
# alpha = 10**(np.sum(x)/n - Sxy/Sxx*np.sum(y)/n)

# print("B-hat =", beta)
# print("log a-hat =", log_alpha)
# print("a-hat =", alpha)
# print(alpha*(300**(beta)))


x = np.array([27, 36, 44, 32, 27, 41, 38, 44, 30, 27, 33, 39, 38, 24, 33, 32, 37, 33, 34, 39])
y = np.array([29, 44, 49, 27, 35, 33, 29, 40, 27, 38, 42, 31, 38, 22, 34, 37, 38, 35, 32, 43])
n=len(x)

Sxy = np.sum(x*y) - (1/n)*(np.sum(x)*np.sum(y))
Sxx = np.sum(x**2) - (1/n)*(np.sum(x))**2
Syy = np.sum(y**2) - (1/n)*(np.sum(y))**2

print("Sxy =", Sxy, "Sxx = ", Sxx, "xbar = ", np.sum(x)/n, "ybar = ", np.sum(y)/n)
beta = Sxy/Sxx
r = Sxy/(Syy*Sxx)**(1/2)
R = (1-r)/(1+r)
z = 1.96
# print(f"r={r}, beta={beta}, Gamma={Gamma}, z={z}")
low = (1-R*np.exp(2*z/np.sqrt(n-3)))/(1+R*np.exp(2*z/np.sqrt(n-3)))
high =(1-R*np.exp(-2*z/np.sqrt(n-3)))/(1+R*np.exp(-2*z/np.sqrt(n-3)))
print(low)
print(high)
