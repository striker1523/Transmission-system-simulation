import math as m
import numpy as np
import matplotlib.pyplot as plt
########################################################################################

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

def fft_split(x):
	FFT = []
	for i in x:
		RE = np.real(i)
		IM = np.imag(i)
		FFT.append([RE, IM])
	return FFT

def ffts(x, N, fs):
	mk = MKfun(x, N)
	mprimk = MprimKfun(mk)
	fk = FK(N, fs)
	return mprimk, fk
########################################################################################
fm = 2
fn = 15
fs = 1000
tc = 1
bufor=fs/tc
n=np.arange(0, bufor-1, 1)
t=n/fs 
mt = np.sin(2*np.pi*fm*t)
#a) 1>ka>0 b) 12>ka>2 c) ka>20
#a) kp<1   b) pi>kp>0 c) kp>2pi
#a) kf<1   b) pi>kf>0 c) kf>2pi

ka, kp, kf = [0.5, 4, 22], [0.3, 2, 9], [0.3, 1, 7]
kat, kpt, kft = ["0.5", "4", "22"], ["0.3", "2", "9"], ["0.3", "1", "7"]

plt.subplot(4,3,2)
plt.plot(mt)
plt.title("Sygnał źródłowy")
iter = 4
for i in range (3):
	zat = (ka[i]*mt+1)*np.cos(2*np.pi*fn*t)
	plt. subplot(4,3,i+iter)
	iter += 1
	plt.plot(zat)
	plt.xlabel("Częstotliwość")
	plt.ylabel("A")

	plt.title("amp ka = " + kat[i])
	zpt = np.cos(2*np.pi*fn*t+kp[i]*mt)
	plt. subplot(4,3,i+iter)
	iter += 1
	plt.plot(zpt)
	plt.xlabel("Częstotliwość")
	plt.ylabel("A")

	plt.title("phase kp = " + kpt[i])
	zft = np.cos(2*np.pi*fn*t+(kf[i]/fm)*mt)
	plt. subplot(4,3,i+iter)
	plt.plot(zft)
	plt.title("freq kf = " + kft[i])
	plt.xlabel("Częstotliwość")
	plt.ylabel("A")

plt.tight_layout()
plt.show()

iter = 1
for i in range (3):
	zat = (ka[i]*mt+1)*np.cos(2*np.pi*fn*t)
	zatf = np.fft.fft(zat)
	N=len(zat)
	x = fft_split(zatf)
	zatmk, zatfk = ffts(x, N, fs)
	plt.subplot(3,3,i+iter)
	iter += 1
	#plt.yscale("log")
	plt.plot(zatfk, zatmk)
	plt.title("Widmo amp ka = " + kat[i])
	plt.xlabel("Częstotliwość")
	plt.ylabel("A")

	zpt = np.cos(2*np.pi*fn*t+kp[i]*mt)
	zptf = np.fft.fft(zpt)
	N=len(zpt)
	x = fft_split(zptf)
	zptmk, zptfk = ffts(x, N, fs)
	plt.subplot(3,3,i+iter)
	iter += 1
	#plt.yscale("log")
	plt.plot(zptfk, zptmk)
	plt.title("Widmo phase kp = " + kat[i])
	plt.xlabel("Częstotliwość")
	plt.ylabel("A")

	zft = np.cos(2*np.pi*fn*t+(kf[i]/fm)*mt)
	zftf = np.fft.fft(zft)
	N=len(zft)
	x = fft_split(zftf)
	zftmk, zftfk = ffts(x, N, fs)
	plt.subplot(3,3,i+iter)
	#plt.yscale("log")
	plt.plot(zftfk, zftmk)
	plt.title("Widmo freq kf = " + kat[i])
	plt.xlabel("Częstotliwość")
	plt.ylabel("A")
plt.tight_layout()
plt.show()

bdb = [3, 6, 12]
ampwidth, phasewidth, freqwidth = [], [], []
for i in range (3):
	zat = (ka[i]*mt+1)*np.cos(2*np.pi*fn*t)
	zatf = np.fft.fft(zat)
	N=len(zat)
	x = fft_split(zatf)
	zatmk, zatfk = ffts(x, N, fs)
	plt. subplot(3,1,1)
	plt.plot(zatfk, zatmk)

	for k in range (3):		#AMP
		width = []
		for j in range (len(zatmk)):
			if zatmk[j] >= (np.max(zatmk) - bdb[k]):
				width.append(zatfk[j])
		ampwidth.append(max(width) - min(width))

	zpt = np.cos(2*np.pi*fn*t+kp[i]*mt)
	zptf = np.fft.fft(zpt)
	N=len(zpt)
	x = fft_split(zptf)
	zptmk, zptfk = ffts(x, N, fs)
	plt. subplot(3,1,2)
	plt.plot(zptfk, zptmk)

	for k in range (3):		#PHASE
		width = []
		for j in range (len(zptmk)):
			if zptmk[j] >= (np.max(zptmk) - bdb[k]):
				width.append(zptfk[j])
		phasewidth.append(max(width) - min(width))

	zft = np.cos(2*np.pi*fn*t+(kf[i]/fm)*mt)
	zftf = np.fft.fft(zft)
	N=len(zft)
	x = fft_split(zftf)
	zftmk, zftfk = ffts(x, N, fs)
	plt. subplot(3,1,3)
	plt.plot(zftfk, zftmk)

	for k in range (3):		#FREQ
		width = []
		for j in range (len(zftmk)):
			if zftmk[j] >= (np.max(zftmk) - bdb[k]):
				width.append(zftfk[j])
		freqwidth.append(max(width) - min(width))
	#plt.show()
byleco=0
for i in range (0, 7, 3):
	byleco +=1
	print("Szerokość modulacji amplitudowej dla",byleco,"sygnału, po kolei B3, B6 I B12:\n",ampwidth[0+i], ampwidth[1+i], ampwidth[2+i])
	print("Szerokość modulacji fazowej dla",byleco,"sygnału, po kolei B3, B6 I B12:\n",phasewidth[0+i], phasewidth[1+i], phasewidth[2+i])
	print("Szerokość modulacji częstotliwości dla",byleco,"sygnału, po kolei B3, B6 I B12:\n",freqwidth[0+i], freqwidth[1+i], freqwidth[2+i],"\n")
