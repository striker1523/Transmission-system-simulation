import math as m
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random

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
# - KODER
def humm74(A):
	if len(A) != 4:
		return 0
	else:
		ret = []
		x1, x2, x3 = (A[0]^A[1]^A[3]), (A[0]^A[2]^A[3]), (A[1]^A[2]^A[3])
		ret.insert(0, x1)
		ret.insert(1, x2)
		ret.insert(2, A[0])
		ret.insert(3, x3)
		ret.insert(4, A[1])
		ret.insert(5, A[2])
		ret.insert(6, A[3])
		return ret
# - MODULACJA
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
# - DEMODULACJA
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
	for i in range (0, len(bits[0])):
		integ = 0
		for j in range (int(len(t))):
			pt.append(integ)
			if jump < len(T):
				integ += key[jump]
				jump += 1
			else:
				break;
	return pt
def CTamp(key):
	ct = []
	for i in range (0, len(key)):
		if key[i] > 75:
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
	for i in range (0, len(ct), int(len(ct)/len(bits[0]))):
		new.append(ct[i:i+int(len(ct)/len(bits[0]))])
	for i in new:
		if np.mean(i) > 0.5:
			slowod.append(1)
		else:
			slowod.append(0)
	return slowod
# - DEKODER
def dehumm74(A):
	x1p, x2p, x3p = (A[2]^A[4]^A[6]), (A[2]^A[5]^A[6]), (A[4]^A[5]^A[6])
	pos = 1
	i = 0
	ret = []
	while pos != 0:
		if i < 2:
			pos=0
			if int(A[0]^x1p != 0):
				pos = pos+1
			if int(A[1]^x2p != 0):
				pos = pos+2
			if int(A[3]^x3p != 0):
				pos = pos+4
			if pos != 0:
				if(A[pos-1] == 1):
					A[pos-1] = 0
				elif(A[pos-1] == 0):
					A[pos-1] = 1
			i += 1
		else:
			return "[E_R_R_O_R]"

	ret.append(A[2])
	ret.append(A[4])
	ret.append(A[5])
	ret.append(A[6])
	return ret
# - BITY
def split_coded_bits(ABC):
	bits = []
	for i in Sbits:
		if len(i) != 4:
			while len(i) !=4:
				i.append(0)
		bits.append(humm74(i))
	return bits
def demode_decode_bits(ABC):
	decode = []
	for i in ABC:
		bit = demodslowo(i)
		zwrot = dehumm74(bit)
		decode.append(zwrot)
	return decode
# - DANE
Slowo = "Transmisja danych"
slowocale = aschange(Slowo)
newbits = slowocale[:100]
Sbits = [newbits[i:i+4] for i in range(0, len(newbits), 4)]
bits = split_coded_bits(Sbits)
print(Sbits)

tc = 1						#---
tb = tc/(len(bits[0]))		
a1 = 1						#+
a2 = 5						#+
w = 2						#
fn = w*(tb**(-1))			
fn1 = (w+1)/tb				
fn2 = (w+2)/tb				
fs = 1000					#---				

t = []
for i in range (int(tb*fs)+1):
	t.append(i/fs)
T = []
for i in range (int(tc*fs)):
	T.append(i/fs)

# - BER
def results(przed, po):
	BER = 0
	for i in range(len(przed)):
		if przed[i] != po[i]:
			BER += 1
	return BER
# - MOD
tamp, tphase, tfreq, bers = [], [], [], []
for i in bits:
	amp = ZAT(t, i)		#Pojedyńcze ramki do modulatora
	phase = ZPT(t, i)		
	freq = ZFT(t, i)	
	amp.pop()			#Nadmiarowa próbka
	phase.pop()
	freq.pop()
	tamp += amp
	tphase += phase
	tfreq += freq

# - SZUM
############################################### ZADANIE 2
bers = []
alpha = 0
for i in range(10):
	tct = []
	non_noised = tfreq							#tamp // tphase // tfreq

	szum = np.random.normal(0, 10, size=len(non_noised))									#SZUM
	szum *= alpha
	alpha += 0.1
	non_noised += szum

	non_noised = [non_noised[i:i+len(T)] for i in range(0, len(non_noised), len(T))]#Rozbicie

	for i in non_noised:
		#xt = XT(i)			#AMP/FAZ
		#pt = PT(xt)		#AMP/FAZ
		#pt.pop()			#AMP/FAZ
		#ct = CTamp(pt)		#DLA AMPLITUDOWEJ
		##ct = CTpsk(pt)	#DLA FAZOWEJ

		xt1, xt2 = FSK(i)	#DLA CZĘSTOTLIWOŚCIOWEJ
		pt1 = PT(xt1)		#DLA CZĘSTOTLIWOŚCIOWEJ
		pt2 = PT(xt2)		#DLA CZĘSTOTLIWOŚCIOWEJ
		pt1.pop()			#DLA CZĘSTOTLIWOŚCIOWEJ
		pt2.pop()			#DLA CZĘSTOTLIWOŚCIOWEJ
		pt = subPT(pt1, pt2)#DLA CZĘSTOTLIWOŚCIOWEJ
		ct = CTfsk(pt)		#DLA CZĘSTOTLIWOŚCIOWEJ

		#plt.plot(T, ct)
		#plt.show()

		tct.append(ct)		#całość ct
	bers.append(results(Sbits, demode_decode_bits(tct)))
	print("Słowo rozszyfrowane:")
	print(demode_decode_bits(tct))
scale = np.arange(0, 1, 0.1)
plt.plot(scale, bers)
plt.ylabel("BER")
plt.xlabel("Alpha")
plt.show()

# - TŁUMIENIE
#bers = []
#for beta in range(0, 10):
#	tct = []
#	non_noised = tfreq							#tamp // tphase // tfreq

#	małet = np.arange(0, len(non_noised), 1)/ len(non_noised)								#TŁUMIENIE
#	gt = np.e**(-beta*małet)																#
#	non_noised *= gt																		#

#	non_noised = [non_noised[i:i+len(T)] for i in range(0, len(non_noised), len(T))]#Rozbicie

#	for i in non_noised:
#		#xt = XT(i)			#AMP/FAZ
#		#pt = PT(xt)		#AMP/FAZ
#		#pt.pop()			#AMP/FAZ
#		##ct = CTamp(pt)	#DLA AMPLITUDOWEJ
#		#ct = CTpsk(pt)		#DLA FAZOWEJ

#		xt1, xt2 = FSK(i)	#DLA CZĘSTOTLIWOŚCIOWEJ
#		pt1 = PT(xt1)		#DLA CZĘSTOTLIWOŚCIOWEJ
#		pt2 = PT(xt2)		#DLA CZĘSTOTLIWOŚCIOWEJ
#		pt1.pop()			#DLA CZĘSTOTLIWOŚCIOWEJ
#		pt2.pop()			#DLA CZĘSTOTLIWOŚCIOWEJ
#		pt = subPT(pt1, pt2)#DLA CZĘSTOTLIWOŚCIOWEJ
#		ct = CTfsk(pt)		#DLA CZĘSTOTLIWOŚCIOWEJ

#		#plt.plot(T, ct)
#		#plt.show()

#		tct.append(ct)		#całość ct
#	bers.append(results(Sbits, demode_decode_bits(tct)))
#	#print("Słowo rozszyfrowane:")
#	#print(demode_decode_bits(tct))
#scale = np.arange(0, 10, 1)
#plt.plot(scale, bers)
#plt.ylabel("BER")
#plt.xlabel("Beta")
#plt.show()
