import math as m
import numpy as np
import matplotlib.pyplot as plt

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

imie = "Bartek"
def aschange(ascii):
	a = []	
	for i in ascii:
		a.append(ord(i))
	ret = []
	for i in a:
		w = i 
		temp = []
		for j in range (8): 
			if (w % 2) == 1:
				w = int(w/2)
				temp.insert(0, 1)
			elif (w % 2) == 0:
				w = int(w/2)
				temp.insert(0, 0)
			else:
				temp.insert(0, 0)
		ret += temp
	return ret
slowocale = aschange(imie)
slowo = slowocale[:10]		    

tc = 1						#---
tb = tc/(len(slowo))		
a1 = 1						#+
a2 = 5						#+
w = 2						#
fn = w*(tb**(-1))			
fn1 = (w+1)/tb				
fn2 = (w+2)/tb				
fs = 1000					#---				

t = []
for i in range (int(tb*fs)):
	t.append(i/fs)
T = []
for i in range (int(tc*fs)):
	T.append(i/fs)

def ZAT(t, key): 
	zat = []
	for i in range (0, len(key)):
		if key[i] == 0:
			for T in t:
				zat.append(a1*np.sin(2*np.pi*fn*T))
		if key[i] == 1:
			for T in t:
				zat.append(a2 * np.sin(2*np.pi*fn*T))
	return zat

def ZPT(t, key): 
	zpt = []
	for i in range (len(key)):
		if key[i] == 0:
			for T in t:
				zpt.append(np.sin(2*np.pi*fn*T))
		if key[i] == 1:
			for T in t:
				zpt.append(np.sin(2*np.pi*fn*T+np.pi))
	return zpt

def ZFT(t, key): 
	zft = []
	for i in range (len(key)):
		if key[i] == 0:
			for T in t:
				zft.append(np.sin(2*np.pi*fn1*T))
		if key[i] == 1:
			for T in t:
				zft.append(np.sin(2*np.pi*fn2*T))
	return zft

amp = ZAT(t, slowo)
phase = ZPT(t, slowo)
freq = ZFT(t, slowo)

plt.subplot(2,3,1)
plt.plot(T, amp)
plt.xlabel("Czas")
plt.ylabel("A")
plt.subplot(2,3,2)
plt.plot(T, phase)
plt.xlabel("Czas")
plt.ylabel("A")
plt.subplot(2,3,3)
plt.plot(T, freq)
plt.xlabel("Czas")
plt.ylabel("A")

tc = 1						#---
tb = tc/(len(slowocale))		
a1 = 1						#+
a2 = 5						#+
w = 2						
fn = w*(tb**(-1))			
fn1 = (w+1)/tb				
fn2 = (w+2)/tb				
fs = 1000					#---				

t = []
for i in range (int(tb*fs)):
	t.append(i/fs)
T = []
for i in range (int(tc*fs)):
	T.append(i/fs)

amp = ZAT(t, slowocale)
phase = ZPT(t, slowocale)
freq = ZFT(t, slowocale)

mamp = np.fft.fft(amp)
mpha = np.fft.fft(phase)
mfrq = np.fft.fft(freq)

N = len(amp)
x = fft_split(mamp)
mampmk, mampfk = ffts(x, N, fs)

N = len(phase)
x = fft_split(mpha)
mphamk, mphafk = ffts(x, N, fs)

N = len(freq)
x = fft_split(mfrq)
mfrqmk, mfrqfk = ffts(x, N, fs)

plt.subplot(2,3,4)
plt.plot(mampfk, mampmk)
plt.ylim(0,30)
plt.title("Widmo kluczowania amplitudowego")
plt.xlabel("Częstotliwość")
plt.ylabel("A")
plt.subplot(2,3,5)
plt.plot(mphafk, mphamk)
plt.ylim(0,30)
plt.title("Widmo kluczowania fazowego")
plt.xlabel("Częstotliwość")
plt.ylabel("A")
plt.subplot(2,3,6)
plt.plot(mfrqfk, mfrqmk)
plt.ylim(-20,30)
plt.title("Widmo kluczowania częstotliwościowego")
plt.xlabel("Częstotliwość")
plt.ylabel("A")
plt.tight_layout()
plt.show()

bdb = [3, 6, 12]
ampwidth, phasewidth, freqwidth = [], [], []
tc = 1						#---
tb = tc/(len(slowocale))		
a1 = 1						#+
a2 = 5						#+
w = 2						
fn = w*(tb**(-1))			
fn1 = (w+1)/tb				
fn2 = (w+2)/tb				
fs = 1000					#---				

t = []
for i in range (int(tb*fs)):
	t.append(i/fs)
T = []
for i in range (int(tc*fs)):
	T.append(i/fs)

amp = ZAT(t, slowocale)
mamp = np.fft.fft(amp)
N = len(amp)
x = fft_split(mamp)
mampmk, mampfk = ffts(x, N, fs)
for k in range (3):
	width = []
	for j in range (len(mampmk)):
		if mampmk[j] >= (np.max(mampmk) - bdb[k]):
			width.append(mampfk[j])
	ampwidth.append(max(width) - min(width))

phase = ZPT(t, slowocale)
mpha = np.fft.fft(phase)
N = len(phase)
x = fft_split(mpha)
mphamk, mphafk = ffts(x, N, fs)
for k in range (3):
	width = []
	for j in range (len(mphamk)):
		if mphamk[j] >= (np.max(mphamk) - bdb[k]):
			width.append(mphafk[j])
	phasewidth.append(max(width) - min(width))

freq = ZFT(t, slowocale)
mfrq = np.fft.fft(freq)
N = len(freq)
x = fft_split(mfrq)
mfrqmk, mfrqfk = ffts(x, N, fs)
for k in range (3):
	width = []
	for j in range (len(mfrqmk)):
		if mfrqmk[j] >= (np.max(mfrqmk) - bdb[k]):
			width.append(mfrqfk[j])
	freqwidth.append(max(width) - min(width))

for i in range (3):
	print("Szerokość modulacji amplitudowej dla",bdb[i],"dB\n",ampwidth[0+i])
	print("Szerokość modulacji fazowej dla",bdb[i],"dB\n",phasewidth[0+i])
	print("Szerokość modulacji częstotliwości dla",bdb[i],"dB\n",freqwidth[0+i],"\n")