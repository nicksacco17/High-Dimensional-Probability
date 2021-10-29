import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import stats

N = 10
NUM_DRAWS = 500
NUM_TEST_POINTS = 100

# X - List of bounded, independent random variables
# m - List of lower bounds on the R.V.'s
# M - List of upper bounds on the R.V.'s
def general_hoeffding_inequality(m, M, t):
    return np.exp((-2 * t**2) / (np.sum((M - m)**2)))

def general_hoeffding_inequality_test(X, t, num_draws):

    count = 0

    for i in range(0, num_draws):
        total = 0
        #if i % (num_draws/10) == 0:
        #    print("ITERATION: %d" % i)
        
        for j in range(0, len(X)):
            total += X[j].rvs() - X[j].mean()
        
        if total >= t:
            count += 1
    return (count/num_draws)

def draw_bernoulli():

    r = stats.bernoulli(0.5)
    print(r.rvs(5))

if __name__ == '__main__':

    X = np.asarray([stats.bernoulli(0.9) for i in range(0, N)])
    m = np.zeros(N, dtype = int)
    M = np.ones(N, dtype = int)

    T = np.linspace(0, 10, NUM_TEST_POINTS)

    prob_upper_bounds_theory = np.zeros(NUM_TEST_POINTS, dtype = float)
    prob_upper_bounds_exper = np.zeros(NUM_TEST_POINTS, dtype = float)
    for index, t in enumerate(T):

        #if index % (len(T)/10) == 0:
        print("TEST POINT %d" % index)
        prob_upper_bounds_theory[index] = general_hoeffding_inequality(m = m, M = M, t = t)
        prob_upper_bounds_exper[index] = general_hoeffding_inequality_test(X = X, t = t, num_draws = NUM_DRAWS)
    
    plt.plot(T, prob_upper_bounds_theory , 'r-', linewidth = 1.5)
    plt.plot(T, prob_upper_bounds_exper , 'b-', linewidth = 1.5)
    plt.show()


    #print(general_hoeffding_inequality(m = m, M = M, t = 5))
    #print(general_hoeffding_inequality_test(X = X, t = 5, num_draws = NUM_DRAWS))


    #fig, ax = plt.subplots(1, 1) 
    #x = np.linspace(stats.norm.ppf(1/N), stats.norm.ppf(1-(1/N)), N)
   
    #r = stats.norm.rvs(size = NUM_DRAWS)

    #ax.plot(x, stats.norm.pdf(x), 'r-', lw = 5, alpha = 0.6, label = 'norm pdf')
    #ax.plot(x, stats.norm.pdf(x), 'k-', lw = 2, label = 'frozen pdf')
    
    #ax.hist(r, density = True, histtype = 'stepfilled', bins = 100, alpha = 0.2)

    #ax.legend(loc = 'best', frameon = False)
    #plt.show()

    #draw_bernoulli()

