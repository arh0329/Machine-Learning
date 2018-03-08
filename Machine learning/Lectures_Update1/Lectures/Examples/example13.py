np.random.seed(1)

n = 20000

x1 = np.random.uniform(0,100,n)
x2 = np.random.uniform(0,10,n)
x3 = np.random.uniform(0,500,n)
x4 = np.random.uniform(0,40,n)
x5 = np.random.uniform(0,2000,n)
x6 = np.random.uniform(0,1000,n)
x7 = np.random.uniform(0,5,n)
x8 = np.random.uniform(0,200,n)
x9 = np.random.uniform(0,2000,n)
x10 = np.random.uniform(0,500,n)

x = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10])
X = x.transpose()
    
xpoly = np.array([x4*x8, x4**4, x7, x8])
Xpoly = xpoly.transpose()

s = 700
y = 0.005 * x4*x8 + .002 * x4**4 + 20 * x7 + 5 * x8 + np.random.normal(0,s,n)