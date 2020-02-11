
# Parameters
theta1 = 0.7
theta2 = 0.5
p1 = 0.01
p2 = 0.01




print("\n\nVALUES UPPER BOUND FORMULATION:")

# Evaluate from 1 to M
M = 10**4
rmin, bmin = M, M

# False negative and false positive conditions
fn_cond = lambda r,b: (1-theta1**r)**b <= p1
fp_cond = lambda r,b: 1-(1-theta2**r)**b <= p2

for r in range(1,M):
	print(f"\rStatus: {r+1}/{M}", end="")
	for b in range(1,M):
		if r*b >= rmin*bmin:  #optimization: no need to evaluate b+1, b+2, ... since r*(b+1) > r*b > rmin*bmin
			break
		if fn_cond(r,b) and fp_cond(r,b):
			rmin = r
			bmin = b

print(f"\nr: {rmin}")
print(f"b: {bmin}")
print(f"Total m: {rmin*bmin}\n")





print("\nVALUES EXACT FORMULATION:")

# Integrate LSH function given r,b from x_from to x_to
def get_integral(r, b, x_from, x_to):
	res = 0
	resolution = 1000
	for i in range(resolution):  #a lot sketchy
		x = x_from + (x_to-x_from)*i/resolution
		res += (1-(1-x**r)**b) * (x_to-x_from)/resolution
	return res


# Loop for all r,b
M = 10**3
rmin, bmin = M, M

# False negative and false positive conditions
fn_cond = lambda r,b: 1-theta1-get_integral(r,b,theta1,1) < p1
fp_cond = lambda r,b: get_integral(r,b,0,theta2) < p2

for r in range(1,M):
	print(f"\rStatus: {r+1}/{M}", end="")
	for b in range(1,M):
		if r*b >= rmin*bmin:  #optimization: no need to evaluate b+1, b+2, ... since r*(b+1) > r*b > rmin*bmin
			break
		if fp_cond(r,b) and fn_cond(r,b):
			rmin = r
			bmin = b


print(f"\nr: {rmin}")
print(f"b: {bmin}")
print(f"Total m: {rmin*bmin}")
