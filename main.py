import sys,os
import yaml

from generator.build_generator import Generator
from ansatz.build_ansatz import Ansatz

mol = sys.argv[1]
cfgpath = 'config/'+mol+'.yml'
respath = 'res/'+mol

with open(cfgpath, 'r', encoding="utf-8") as file:
    cfg = yaml.load(file.read())
    file.close()
   
os.makedirs(respath, exist_ok=True)
if cfg['setting']['log'] is True:
    sys.stdout = open(respath+'/res.log', 'w')


generator = Generator(cfg)
ansatz = Ansatz(cfg)

hamiltonian = generator.run()
result = ansatz.run(hamiltonian, generator.n_qubits)