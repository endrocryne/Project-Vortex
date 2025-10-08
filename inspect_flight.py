import inspect
import rocketpy.simulation.flight as rf
from rocketpy import Flight

print('Flight.__init__ signature:')
print(inspect.signature(Flight.__init__))
print('\nFlight module file:')
print(rf.__file__)

src = inspect.getsource(rf)
lines = src.splitlines()
print('\nSearching for assignments to self.atol...')
for i,line in enumerate(lines):
    if 'self.atol' in line or 'atol=' in line:
        start = max(0, i-3)
        end = min(len(lines), i+3)
        print('\n--- around line', i+1, '---')
        for j in range(start, end):
            print(f'{j+1:5d}: {lines[j]}')

# Print Flight.__init__ source for review
print('\n--- Flight.__init__ (first 300 lines) ---')
init_src = inspect.getsource(Flight.__init__)
print('\n'.join(init_src.splitlines()[:300]))
