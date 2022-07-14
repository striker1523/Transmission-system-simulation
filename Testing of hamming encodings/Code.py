import math as m
import numpy as np
import matplotlib.pyplot as plt


#Kod (7,4)
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
			return "Słowo posiada więcej niż jeden bład."

	ret.append(A[2])
	ret.append(A[4])
	ret.append(A[5])
	ret.append(A[6])
	return ret

#A = [1, 1, 0, 0]
#wynik1 = humm74(A)
#print(wynik1)
#B = [0, 1, 1, 1, 1, 0, 0]
#wynik2 = dehumm74(B)
#print(wynik2)

#Kod (15,11)
def humm1511(B):
	P = np.array([[1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
				 [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
				 [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
				 [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]],
			  )
	P = P.transpose()
	I = np.eye(11)
	G = np.hstack((P, I))
	c = np.dot(B, G)
	for i in range (0, 4):
		c[i] = c[i] % 2
	return c

def dehumm1511(C):
	P = np.array([[1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
				[1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
				[0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
				[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]],
			)
	I = np.eye(len(C)-11)
	H = np.hstack((I, P))
	H = H.transpose()

	s = np.dot(C, H)
	for i in range (0, 4):
		s[i] = s[i] % 2
	i = 0
	S = 0
	while (i < 2):
		for i in range (0, 4):
			S += s[i] * 2 ** i
		print(S)
		if S != 0:
			if(C[int(S-1)] == 1):
				C[int(S-1)] = 0
			elif(C[int(S-1)] == 0):
				C[int(S-1)] = 1
			i += 1
		elif S == 0:
			return C[4:15]
		elif S > 15:
			return "Słowo posiada więcej niż jeden bład."
		else:
			return "Słowo posiada więcej niż jeden bład."
		S = 0
	return C[4:15]

slowo = [ 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0 ]
		#							 \1/-10   \1/-13
deslowo = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0 ]

print(humm1511(slowo))
print(dehumm1511(deslowo))