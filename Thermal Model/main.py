import finite_difference as fd

def main():
    solver = fd.FiniteDifference_1D(40, 0.0105, 700, 1, 4030, 815, 1, 5, 298.15, 300000, 298.15)
    solver.BackwardEuler()
    solver.Plot()

if __name__ == "__main__":
    main()
