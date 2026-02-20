# ramanforge
simulate raman spectrum using the Born effective charges method.

install with 
```
pip install .
```
usage guide:
```
from BORNRaman.calculator import Raman, Phonon
from .util import init_mace, init_E3NN

atoms = read('path_to_struct')
phonon = Phonon(atoms, calc = init_mace())
evecs, evals = phonon.calculate()
raman = Raman(atoms, calc = init_E3NN, evecs = evecs, evals=evals)
raman.calculate(quantity = 'dq', dE=0.1)
raman.calculate(quantity = 'activity')
raman.plot(x_min =300, x_max=500)
```
