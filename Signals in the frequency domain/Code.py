import math as m
import numpy as np
import matplotlib.pyplot as plt

def DFTfun(x, K, N):
	XK = []								#DFT
	XKRE, XKIM = 0, 0
	for k in range (len(K)):
		for n in range (len(K)):
			XKRE += x[n] * (np.cos((-2*np.pi*k*n)/N))
			XKIM += x[n] * (np.sin((-2*np.pi*k*n)/N))
		XK.append([XKRE, XKIM])
		XKRE, XKIM = 0, 0
	return XK
def MKfun(XK, N):
	MK = []								#MK
	for i in range (0, int(N/2)):
		MK.append(np.sqrt(XK[i][0]**2 + XK[i][1]**2))
	return MK
def MprimKfun(MK):
	MprimK = []							#M'K
	for i in (MK):
		MprimK.append(10*np.log10(i))
	return MprimK
def FK(N, fs):
	FK = []
	for k in range (0, int(N/2)):
		FK.append(k*(fs/(N/2)))
	return FK
############################################################################
#DANE
#N=512								
#fs=1000								
#t=1
#K = np.arange (0, N, 1)					
#x=np.sin(200*np.pi*t*(K/fs))				

#xk = DFTfun(x, K, N)
#mk = MKfun(xk, N)
#mprimk = MprimKfun(mk)
#fk = FK(N, fs)
#	#D									
#plt.plot(fk, mprimk)
#plt.show()


N=512									
fs=1000									
f=250									
tc=1
bufor=fs*tc								
n=np.arange(0, bufor-1, 1)			
t=n/fs									
K = np.arange (0, N, 1)					

#xt
xt=np.sin(np.absolute(np.sin(2*np.pi*f*t**2)**13)+np.cos(2*np.pi*t))		
xk = DFTfun(xt, K, N)
mk = MKfun(xk, N)
mprimk = MprimKfun(mk)
fk = FK(N, fs)
plt.subplot(4,2,1)
plt.plot(fk, mprimk)
plt.title("Widmo dla xt")
plt.xlabel("Częstotliwość")
plt.ylabel("A")

yt=(xt*t**3)/3			
xk = DFTfun(yt, K, N)
mk = MKfun(xk, N)
mprimk = MprimKfun(mk)
fk = FK(N, fs)
plt.subplot(4,2,2)
plt.plot(fk, mprimk)
plt.title("Widmo dla yt")
plt.xlabel("Częstotliwość")
plt.ylabel("A")

zt = 1.92*(np.cos(3*np.pi*t/2)+np.cos(yt**2/(8*xt+3)*t))														
xk = DFTfun(zt, K, N)
mk = MKfun(xk, N)
mprimk = MprimKfun(mk)
fk = FK(N, fs)
plt.subplot(4,2,3)
plt.plot(fk, mprimk)
plt.title("Widmo dla zt")
plt.xlabel("Częstotliwość")
plt.ylabel("A")

vt = (yt*zt/(xt+2))*np.cos(7.2*np.pi*t)+np.sin(np.pi*t**2)													
xk = DFTfun(vt, K, N)
mk = MKfun(xk, N)
mprimk = MprimKfun(mk)
fk = FK(N, fs)
plt.subplot(4,2,4)
plt.plot(fk, mprimk)
plt.title("Widmo dla vt")
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
xk = DFTfun(ut, K, N)
mk = MKfun(xk, N)
mprimk = MprimKfun(mk)
fk = FK(N, fs)
plt.subplot(4,2,5)
plt.plot(fk, mprimk)
plt.title("Widmo dla ut")
plt.xlabel("Częstotliwość")
plt.ylabel("A")

h1, h2, h3 = 2, 5, 25
b1t, b2t, b3t = 0, 0, 0
for h in range(h1):
	b1t = b1t + (np.sin(np.sin(np.pi*h/7*t)*np.pi*t*h))/(2*h**2+1)
for h in range(h2):
	b2t = b2t + (np.sin(np.sin(np.pi*h/7*t)*np.pi*t*h))/(2*h**2+1)
for h in range(h3):
	b3t = b3t + (np.sin(np.sin(np.pi*h/7*t)*np.pi*t*h))/(2*h**2+1)
xk = DFTfun(b1t, K, N)
mk = MKfun(xk, N)
mprimk = MprimKfun(mk)
fk = FK(N, fs)
plt.subplot(4,2,6)
plt.plot(fk, mprimk)
plt.title("Widmo dla b1t")
plt.xlabel("Częstotliwość")
plt.ylabel("A")

xk = DFTfun(b2t, K, N)
mk = MKfun(xk, N)
mprimk = MprimKfun(mk)
fk = FK(N, fs)
plt.subplot(4,2,7)
plt.plot(fk, mprimk)
plt.title("Widmo dla b2t")
plt.xlabel("Częstotliwość")
plt.ylabel("A")


xk = DFTfun(b3t, K, N)
mk = MKfun(xk, N)
mprimk = MprimKfun(mk)
fk = FK(N, fs)
plt.subplot(4,2,8)
plt.plot(fk, mprimk)
plt.title("Widmo dla b3t")
plt.xlabel("Częstotliwość")
plt.ylabel("A")
plt.tight_layout()
plt.show()

###############################################################################
#N=2048									
#fs=10000								
#f=5000									
#tc=2
#bufor=fs*tc							
#n=np.arange(0, bufor-1, 1)				
#t=n/fs									
#K = np.arange (0, N, 1)					

##fs=22005
##f=10000
##tc=1
##BUFOR=fs*tc 
##n=np.arange(0, BUFOR-1, 1) 
##t=n/fs

#def fft_split(x):
#	FFT = []
#	for i in x:
#		RE = np.real(i)
#		IM = np.imag(i)
#		FFT.append([RE, IM])
#	return FFT

#import time
#def dfts(x, K, N, fs):
#	xk = DFTfun(x, K, N)
#	mk = MKfun(xk, N)
#	mprimk = MprimKfun(mk)
#	fk = FK(N, fs)
#	return mprimk, fk
#def ffts(x, N, fs):
#	mk = MKfun(x, N)
#	mprimk = MprimKfun(mk)
#	fk = FK(N, fs)
#	return mprimk, fk

#timesd, timesf = [], []

#xt = np.sin(np.absolute(np.sin(2*np.pi*f*t**2)**13)+np.cos(2*np.pi*t))
#x = xt

#dft1 = time.time()
#mprimk, fk = dfts(x, K, N, fs)
#dft2 = time.time()
#print(dft2 - dft1)
#timesd.append(dft2-dft1)

#fft1 = time.time()
#fx = np.fft.fft(x)
#fftx = fft_split(fx)
#mprimk, fk = ffts(fftx, N, fs)
#fft2 = time.time()

#print(fft2 - fft1)
#timesf.append(fft2-fft1)

#yt = (xt*t**3)/3
#x = yt

#dft1 = time.time()
#mprimk, fk = dfts(x, K, N, fs)
#dft2 = time.time()
#print(dft2 - dft1)
#timesd.append(dft2-dft1)

#fft1 = time.time()
#fx = np.fft.fft(x)
#fftx = fft_split(fx)
#mprimk, fk = ffts(fftx, N, fs)
#fft2 = time.time()

#print(fft2 - fft1)
#timesf.append(fft2-fft1)

#zt = 1.92*(np.cos(3*np.pi*t/2)+np.cos(yt**2/(8*xt+3)*t))
#x = zt

#dft1 = time.time()
#mprimk, fk = dfts(x, K, N, fs)
#dft2 = time.time()
#print(dft2 - dft1)
#timesd.append(dft2-dft1)

#fft1 = time.time()
#fx = np.fft.fft(x)
#fftx = fft_split(fx)
#mprimk, fk = ffts(fftx, N, fs)
#fft2 = time.time()

#print(fft2 - fft1)
#timesf.append(fft2-fft1)

#vt = (yt*zt/(xt+2))*np.cos(7.2*np.pi*t)+np.sin(np.pi*t**2)
#x = vt

#dft1 = time.time()
#mprimk, fk = dfts(x, K, N, fs)
#dft2 = time.time()
#print(dft2 - dft1)
#timesd.append(dft2-dft1)

#fft1 = time.time()
#fx = np.fft.fft(x)
#fftx = fft_split(fx)
#mprimk, fk = ffts(fftx, N, fs)
#fft2 = time.time()

#print(fft2 - fft1)
#timesf.append(fft2-fft1)

#ut = []
#def funkcja(t):
#	if(i < 0.3 and i >= 0):
#		return 1/2*np.absolute(np.cos(3*np.pi*t)*np.sin(2.2*np.pi*t**2))**0.32
#	elif(i < 1 and i >= 0.3):
#		return 1.1*t*((np.cos(10*np.pi*t-np.pi))/(np.sin(np.pi*t**2)+4))
#	elif(i < 2 and i >= 1):
#		return (np.absolute((t+1)*np.sin(8*t**2+np.pi/2+0.14)**3)/(8.6))
#	elif(i < 2.6 and i >= 2):
#		return (t**4*np.log10(t))/30
#for i in t:
#	ut.append(funkcja(i))
#x = ut

#dft1 = time.time()
#mprimk, fk = dfts(x, K, N, fs)
#dft2 = time.time()
#print(dft2 - dft1)
#timesd.append(dft2-dft1)

#fft1 = time.time()
#fx = np.fft.fft(x)
#fftx = fft_split(fx)
#mprimk, fk = ffts(fftx, N, fs)
#fft2 = time.time()

#print(fft2 - fft1)
#timesf.append(fft2-fft1)

#fs=22005
#f=10000 
#tc=1
#BUFOR=fs*tc 
#n=np.arange(0, BUFOR-1, 1) 
#t=n/fs
#h1, h2, h3 = 2, 5, 25
#b1t, b2t, b3t = 0, 0, 0
#for h in range(h1):
#	b1t = b1t + (np.sin(np.sin(np.pi*h/7*t)*np.pi*t*h))/(2*h**2+1)
#x = b1t

#dft1 = time.time()
#mprimk, fk = dfts(x, K, N, fs)
#dft2 = time.time()
#print(dft2 - dft1)
#timesd.append(dft2-dft1)

#fft1 = time.time()
#fx = np.fft.fft(x)
#fftx = fft_split(fx)
#mprimk, fk = ffts(fftx, N, fs)
#fft2 = time.time()

#print(fft2 - fft1)
#timesf.append(fft2-fft1)

#for h in range(h2):
#	b2t = b2t + (np.sin(np.sin(np.pi*h/7*t)*np.pi*t*h))/(2*h**2+1)
#x = b2t

#dft1 = time.time()
#mprimk, fk = dfts(x, K, N, fs)
#dft2 = time.time()
#print(dft2 - dft1)
#timesd.append(dft2-dft1)

#fft1 = time.time()
#fx = np.fft.fft(x)
#fftx = fft_split(fx)
#mprimk, fk = ffts(fftx, N, fs)
#fft2 = time.time()

#print(fft2 - fft1)
#timesf.append(fft2-fft1)

#for h in range(h3):
#	b3t = b3t + (np.sin(np.sin(np.pi*h/7*t)*np.pi*t*h))/(2*h**2+1)
#x = b3t

#dft1 = time.time()
#mprimk, fk = dfts(x, K, N, fs)
#dft2 = time.time()
#print(dft2 - dft1)
#timesd.append(dft2-dft1)

#fft1 = time.time()
#fx = np.fft.fft(x)
#fftx = fft_split(fx)
#mprimk, fk = ffts(fftx, N, fs)
#fft2 = time.time()

#print(fft2 - fft1)
#timesf.append(fft2-fft1)

#print("Średni czas DFT: ",sum(timesd)/8)
#print("Średni czas FFT: ",sum(timesf)/8)