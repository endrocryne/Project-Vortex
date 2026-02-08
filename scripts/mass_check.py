import csv
with open('../results/single_run_20260128_163835/single_run.csv') as f:
    r=csv.reader(f)
    next(r)
    for i,row in enumerate(r):
        t=float(row[0]); mass=float(row[11])
        if mass < 59.999:
            print('first mass drop at line', i+2, 'time', t, 'mass', mass)
            break
print('done')
