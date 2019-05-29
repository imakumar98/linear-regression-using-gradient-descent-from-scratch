from numpy import *


def run():
    points = genfromtxt('data.csv', delimiter=',')
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print(b)
    print(m)


def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations):
    b = initial_b
    m = initial_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]


def step_gradient(b, m, points, learning_rate):

    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]

        b_gradient += -(2/N) * (y-((m*x)+b))
        m_gradient += -(2/N) * x * (y-((m*x) + b))
    new_b = b - (learning_rate*b_gradient)
    new_m = m - (learning_rate*m_gradient)
    return [new_b, new_m]





if __name__ == '__main__':
    run()
