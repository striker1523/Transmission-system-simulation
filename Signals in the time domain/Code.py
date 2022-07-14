import math as m
import numpy as np
import matplotlib.pyplot as plt

fs=10000
f=1000 
tc=2
bufor=fs*tc 
n=np.arange(0, bufor-1, 1) 
t=n/fs

xt = np.absolute(np.sin(2*np.pi*f*t**2)**13)+np.cos(2*np.pi*t)
plt.subplot(4,1,1)
plt.plot(xt)
plt.xlabel("Częstotliwość")
plt.ylabel("A")

yt = (xt*t**3)/3
zt = 1.92*(np.cos(3*np.pi*t/2)+np.cos(yt**2/(8*xt+3)*t))
vt = (yt*zt/(xt+2))*np.cos(7.2*np.pi*t)+np.sin(np.pi*t**2)
plt.subplot(4,3,4)
plt.plot(yt)
plt.xlabel("Częstotliwość")
plt.ylabel("A")
plt.subplot(4,3,5)
plt.plot(zt)
plt.xlabel("Częstotliwość")
plt.ylabel("A")
plt.subplot(4,3,6)
plt.plot(vt)
plt.xlabel("Częstotliwość")
plt.ylabel("A")

ut = []
def funkcja(t):
	if(i < 0.3 and i >= 0):
		return 1/2*np.absolute(np.cos(3*np.pi*t)*np.sin(2.2*np.pi*t**2))**0.32
	elif(i < 1 and i >= 0.3):
		return 1.1*t*((np.cos(10*np.pi*t-np.pi))/(np.sin(np.pi*t**2)+4))
	elif(i < 2 and i >= 1):
		return (np.absolute((t+1)*np.sin(8*t**2+np.pi/2+0.14)**3)/(8.6))
	elif(i < 2.6 and i >= 2):
		return (t**4*np.log10(t))/30
for i in t:
	ut.append(funkcja(i))
plt.subplot(4,1,3)
plt.plot(ut)
plt.xlabel("Częstotliwość")
plt.ylabel("A")

FS=22005
F=10000 
TC=1
BUFOR=FS*TC 
N=np.arange(0, BUFOR-1, 1) 
T=N/FS 
h1, h2, h3 = 2, 5, 25
b1t, b2t, b3t = 0, 0, 0
for h in range(h1):
	b1t = b1t + (np.sin(np.sin(np.pi*h/7*T)*np.pi*T*h))/(2*h**2+1)
for h in range(h2):
	b2t = b2t + (np.sin(np.sin(np.pi*h/7*T)*np.pi*T*h))/(2*h**2+1)
for h in range(h3):
	b3t = b3t + (np.sin(np.sin(np.pi*h/7*T)*np.pi*T*h))/(2*h**2+1)
plt.subplot(4,3,10)
plt.plot(b1t)
plt.xlabel("Częstotliwość")
plt.ylabel("A")
plt.subplot(4,3,11)
plt.plot(b2t)
plt.title("Zestaw funkcji numer 2")
plt.xlabel("Częstotliwość")
plt.ylabel("A")
plt.subplot(4,3,12)
plt.plot(b3t)
plt.title("Zestaw funkcji numer 2")
plt.xlabel("Częstotliwość")
plt.ylabel("A")
plt.tight_layout()
plt.show()