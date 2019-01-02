import numpy as np

sx = 2
sy = 2
B = 3
C = 4

probs = np.zeros((2,sx,sy,B*C))
probs = np.reshape(probs,(2,sx,sy,B,C))
probs[0][0][1][0][1] = 1.

prob_ = np.full((2,sx,sy,B,C), 0.5)

obj = np.zeros((2,sx,sy,B,1))
obj[0][0][1][0] = 1.
obj[1][0][1][0] = 1.

print(probs)
print('___________')
print(prob_)
print('___________')
print(obj)
print(probs.shape, prob_.shape, obj.shape)

subP = probs - prob_
lossP = np.sum(obj * (np.square(subP)))

print(lossP)
