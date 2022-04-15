import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from functools import lru_cache
from pandas import DataFrame


def save(filename, results):
  filename += '.xlsx'
  df = DataFrame(data=results)
  print(df)
  df.to_csv(filename, index=False, header=False)

class quadraticSpline():

  def __init__(self, X, Y, bnd_cond):
    self.functions = []
    self.X = X
    self.Y = Y
    self.boudary_condition = bnd_cond
    self.solve()

  def get_interval(self, x):
    n = len(self.X)
    l = 0
    r = n - 1

    while l <= r:
      m = (l + r) // 2
      if x >= self.X[m]:
        l = m + 1
      else:
        r = m - 1
    return l - 1

  def gamma(self, i):
    return (self.Y[i] - self.Y[i - 1]) / (self.X[i] - self.X[i - 1])

  def a_natural(self, i):
    return (self.b_natural(i + 1) - self.b_natural(i)) / (2 * (self.X[i + 1] - self.X[i]))

  @lru_cache(maxsize=None)
  def b_natural(self, i):
    if i == 0: return 0 # WARUNEK BRZEGOWY
    return 2 * self.gamma(i) - self.b_natural(i-1)

  def a_clamped(self, i):
    return (self.b_clamped(i + 1) - self.b_clamped(i)) / (2 * (self.X[i + 1] - self.X[i]))

  @lru_cache(maxsize=None)
  def b_clamped(self, i):
    if i == 0:
      return (self.Y[1] - self.Y[0]) / (self.X[1] - self.X[0])  # WARUNEK BRZEGOWY
    else:
      return 2 * self.gamma(i) - self.b_clamped(i-1)

  def solve(self):
    if self.boudary_condition == 'natural':
      self.calc_functions(self.a_natural, self.b_natural)
    elif self.boudary_condition == "clamped":
      self.calc_functions(self.a_clamped, self.b_clamped)

  def calc_functions(self, a, b):
    n = len(self.X)
    def s(i):
      def f(x):
        diff = (x - self.X[i])
        return a(i) * diff**2 + b(i) * diff + self.Y[i]
      return f

    for i in range(n-1):
      self.functions.append(s(i))

  def S(self, xs):
    n = len(self.X)
    output = [self.functions[max(0, min(self.get_interval(x), n-2))](x) for x in xs]
    return output


class CubicSpline():
  def __init__(self, X, Y, bnd_cond):
    self.X = X
    self.Y = Y
    self.boundary_conditions = bnd_cond
    self.foos = []
    self.sigmas = None
    self.solve()


  def h(self, i):
    return self.X[i+1] - self.X[i]

  def delta(self, i):
    return (self.Y[i+1] - self.Y[i]) / self.h(i)

  def delta2(self, i):
    return (self.delta(i+1) - self.delta(i)) / (self.X[i + 1] - self.X[i-1])

  def delta3(self, i):
    return (self.delta2(i+1) - self.delta2(i)) / (self.X[i+2] - self.X[i-1])

  def solve(self):
    n = len(self.X)
    h_matrix = [[0 for _ in range(n)] for __ in range(n)]
    d_matrix = [0 for _ in range(n)]

    # tutaj dzieje się n-1 równań
    for i in range(1, n-1):
      h_matrix[i][i-1] = self.h(i-1)
      h_matrix[i][i] = 2 * (self.h(i-1) + self.h(i))
      h_matrix[i][i+1] = self.h(i)

      d_matrix[i] = self.delta(i) - self.delta(i-1)

    # apply bnd condition TUTAJ SĄ BOUNDARY CONDITIONS
    self.apply_boundary_conditions(h_matrix, d_matrix)

    self.calc_functions()

  def apply_boundary_conditions(self, h_matrix, d_matrix):
    n = len(self.X)
    if self.boundary_conditions == "cubic":
      h_matrix[0][0] = -self.h(0)
      h_matrix[0][1] = self.h(0)
      h_matrix[n-1][n-2] = self.h(n-2)
      h_matrix[n-1][n-1] = -self.h(n-2)

      d_matrix[0] = self.h(0)**2 * self.delta3(0)
      d_matrix[n-1] = -self.h(n-2)**2 * self.delta3(n-4)

      self.sigmas = np.linalg.solve(h_matrix, d_matrix)

    elif self.boundary_conditions == 'natural':
      h_matrix = [row[1:-1] for row in h_matrix[1:-1]]
      d_matrix = d_matrix[1:-1]

      self.sigmas = [0, *np.linalg.solve(h_matrix, d_matrix), 0]


  def get_interval(self, x):
    n = len(self.X)
    l = 0
    r = n - 1

    while l <= r:
      m = (l+r) // 2
      if x >= self.X[m]:
        l = m + 1
      else:
        r = m - 1
    return l-1

  def b(self, i):
    return (self.Y[i+1] - self.Y[i]) / self.h(i) -\
           self.h(i) * (self.sigmas[i+1] + 2*self.sigmas[i])

  def c(self, i):
    return 3 * self.sigmas[i]

  def d(self, i):
    return (self.sigmas[i+1] - self.sigmas[i]) / self.h(i)

  def calc_functions(self):
    n = len(self.X)
    for i in range(n-1):
      def s(i):
        def f(x):
          diff = x - self.X[i]
          return self.Y[i] + \
                 self.b(i) * diff + \
                 self.c(i) * diff ** 2 + \
                 self.d(i) * diff ** 3
        return f
      self.foos.append((s(i)))

  def S(self, xs):
    n = len(self.X)
    output = [self.foos[max(0, min(self.get_interval(x), n - 2))](x) for x in xs]
    return output


def plot(space, *functions, points=None, title=None):
  plt.rcParams['figure.figsize'] = [9, 6]

  if points != None:
    plt.scatter(points[0], points[1], label="nodes")

  for foo, lbl, line in functions:
    plt.plot(space, foo(space), line, label=lbl)

  if title:
    plt.title(title)

  plt.legend(bbox_to_anchor=(0.85, 0.23), loc='upper left', borderaxespad=0)
  plt.grid()
  plt.show()

def get_norm(y1, y2, mode):
  n = len(y1)
  if mode == 'max':
    return max([abs(y1[i] - y2[i]) for i in range(n)])
  elif mode == 'sse':
    return sum([(y1[i] - y2[i])**2 for i in range(n)])

k = 1
m = 2
def f(x):
    return -k * x * math.sin(m*(x-1))
f = np.vectorize(f)
a = -3 * math.pi + 1
b = 2 * math.pi + 1
x_space = np.linspace(a, b, 1000)
ys_original = f(x_space)

res = [['Liczba węzłów', 'spl2, nat', 'spl2, clamped', 'spl3, nat', 'spl3, cub']]


for n in [4, 5, 8, 10, 12, 14, 18, 20, 25, 30, 40, 50, 60, 70, 80, 100]:
  X = np.linspace(a, b, n)
  Y = f(X)

  r = [n]

  cubic_spline_cubic = CubicSpline(X, Y, 'cubic')
  cubic_spline_natural = CubicSpline(X, Y, 'natural')
  quadratic_spline_natural = quadraticSpline(X, Y, 'natural')
  quadratic_spline_clamped = quadraticSpline(X, Y, "clamped")

  r.append(get_norm(quadratic_spline_natural.S(x_space), ys_original, "sse"))
  r.append(get_norm(quadratic_spline_clamped.S(x_space), ys_original, "sse"))
  r.append(get_norm(cubic_spline_natural.S(x_space), ys_original, "sse"))
  r.append(get_norm(cubic_spline_cubic.S(x_space), ys_original, "sse"))
  res.append(r)

  plot(x_space,
       [cubic_spline_cubic.S, "CUBIC 3rd", 'g-'],
       [cubic_spline_natural.S, "NATURAL 3rd", 'r-.'],
       [quadratic_spline_natural.S, "NATURAL 2nd", 'y-'],
       [quadratic_spline_clamped.S, "CLAMPED 2nd", 'b-.'],
       [f, "f(x)", 'm-'],
       points=[X, Y], title=f"n = {n}")

save('res_eq', res)
