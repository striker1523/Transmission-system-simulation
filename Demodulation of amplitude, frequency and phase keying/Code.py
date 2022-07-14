import math as m
import numpy as np
import matplotlib.pyplot as plt

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
tb = tc/(len(slowo))		#
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

def ZAT(t, key): #Kluczowanie amplitudy
	zat = []
	for i in range (0, len(key)):
		if key[i] == 0:
			for T in t:
				zat.append(a1*np.sin(2*np.pi*fn*T))
		if key[i] == 1:
			for T in t:
				zat.append(a2 * np.sin(2*np.pi*fn*T))
	return zat
def ZPT(t, key): #Kluczowanie fazy
	zpt = []
	for i in range (len(key)):
		if key[i] == 0:
			for T in t:
				zpt.append(np.sin(2*np.pi*fn*T))
		if key[i] == 1:
			for T in t:
				zpt.append(np.sin(2*np.pi*fn*T+np.pi))
	return zpt

def ZFT(t, key): #Kluczowanie częstotliwości
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

def XT(key):
	xt = []
	for i in range (0, len(key)):
		demod = (a1*np.sin(2*np.pi*fn*T[i]))
		xt.append(key[i]*demod)
	return xt

def PT(key):
	pt = []
	htemp = []
	jump = 0
	for i in range (0, len(slowo)):
		integ = 0
		for j in range (int(fs*0.1)):
			pt.append(integ)
			integ += key[jump]
			jump += 1
		htemp.append(integ)
	h = min(htemp)
	return pt, h

def CTamp(key, h):
	ct = []
	for i in range (0, len(key)):
		if key[i] > h:
			ct.append(1)
		else:
			ct.append(0)
	return ct

def CTpsk(key):
	ct = []
	for i in range (0, len(key)):
		if key[i] < 0:
			ct.append(1)
		else:
			ct.append(0)
	return ct

def CTfsk(key):
	ct = []
	for i in range (0, len(key)):
		if key[i] > 0:
			ct.append(1)
		else:
			ct.append(0)
	return ct

def FSK(key):
	xt1 = []
	xt2 = []
	for i in range (0, len(key)):
		xt1.append(key[i]*(a1*np.sin(2*np.pi*fn1*T[i])))
		xt2.append(key[i]*(a1*np.sin(2*np.pi*fn2*T[i])))
	return xt1, xt2

def subPT(pt1, pt2):
	pt = []
	for i in range (fs):
		pt.append(pt2[i] - pt1[i])
	return pt

def demodslowo(ct):
	new = []
	slowod = []
	for i in range (0, len(ct), int(len(ct)/len(slowo))):
		new.append(ct[i:i+int(len(ct)/len(slowo))])
	for i in new:
		if np.mean(i) > 0.5:
			slowod.append(1)
		else:
			slowod.append(0)
	return slowod

###################AMP
plt.subplot(4,1,1)
plt.plot(T, amp)
plt.title("Sygnał zmodulowany amplitudowo")
plt.xlabel("Czas")
plt.ylabel("A")
xt = XT(amp)
plt.subplot(4,1,2)
plt.plot(T, xt)
plt.title("Demodulacja xt")
plt.xlabel("Czas")
plt.ylabel("A")
pt, h = PT(xt)
plt.subplot(4,1,3)
plt.plot(T, pt)
plt.title("Demodulacja pt")
plt.xlabel("Czas")
plt.ylabel("A")
ct = CTamp(pt, h)
plt.subplot(4,1,4)
plt.plot(T, ct)
plt.title("Demodulacja ct")
plt.xlabel("Czas")
plt.ylabel("A")
demod = demodslowo(ct)
print("Słowo rozszyfrowane z kluczowania amplitudy:", demod)
plt.tight_layout()
plt.show()

##################PHASE
plt.subplot(4,1,1)
plt.plot(T, phase)
plt.title("Sygnał zmodulowany fazowo")
plt.xlabel("Czas")
plt.ylabel("A")
xt = XT(phase)
plt.subplot(4,1,2)
plt.plot(T, xt)
plt.title("Demodulacja xt")
plt.xlabel("Czas")
plt.ylabel("A")
pt, h = PT(xt)
plt.subplot(4,1,3)
plt.plot(T, pt)
plt.title("Demodulacja pt")
plt.xlabel("Czas")
plt.ylabel("A")
ct = CTpsk(pt)
plt.subplot(4,1,4)
plt.plot(T, ct)
plt.title("Demodulacja ct")
plt.xlabel("Czas")
plt.ylabel("A")
demod = demodslowo(ct)
print("Słowo rozszyfrowane z kluczowania fazy:", demod)
plt.tight_layout()
plt.show()

##################FREQ
plt.subplot(6,1,1)
plt.plot(T, freq)
plt.title("Sygnał zmodulowany częstotliwościowo")
plt.xlabel("Czas")
plt.ylabel("A")
xt1, xt2 = FSK(freq)
plt.subplot(6,1,2)
plt.plot(T, xt1)
plt.title("Demodulacja xt1")
plt.xlabel("Czas")
plt.ylabel("A")
plt.subplot(6,1,3)
plt.plot(T, xt2)
plt.title("Demodulacja xt2")
plt.xlabel("Czas")
plt.ylabel("A")
pt1, h = PT(xt1)
plt.subplot(6,1,4)
plt.plot(T, pt1)
plt.title("Demodulacja pt1")
plt.xlabel("Czas")
plt.ylabel("A")
pt2, h = PT(xt2)
plt.subplot(6,1,5)
plt.plot(T, pt2)
plt.title("Demodulacja pt2")
plt.xlabel("Czas")
plt.ylabel("A")
pt = subPT(pt1, pt2)
ct = CTfsk(pt)
plt.subplot(6,1,6)
plt.plot(T, ct)
plt.title("Demodulacja ct")
plt.xlabel("Czas")
plt.ylabel("A")
demod = demodslowo(ct)
print("Słowo rozszyfrowane z kluczowania częstotliwości:", demod)
plt.tight_layout()
plt.show()