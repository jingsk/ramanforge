from becderqspr.calculator import calculate_becder
import torch
import numpy as np
from ase.units import invcm
import logging

default_dtype = torch.float32
torch.set_default_dtype(default_dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#unit: e^2/(eV*A)
epslon_0 = 55.2635e-4

class Raman:
    '''
    Calculator for raman properties using precalculated eigenvector, eigenvalues. 
    '''

    def __init__(atoms, calculator, eigenvector, eigenvalues):
        '''
        construct a Raman class.
        '''
        self.atoms = atoms
        self.calculator = calculator
        self.eigenvector = eigenvector.real
        self.eigenvalues = eigenvalues
        self.dq = None
        self.activity = None
    
    def _calculate_dq(self, dE):
        becder = calculate_becder(self.atoms, self.calculator)
        self.dq = becder
    
    def _calculate_activity(self, dE):
        '''
        Originally written by Author: Rui Zhang Date: 04/28/2025
        Published in Zhang, Rui, et al. "RASCBEC: Raman spectroscopy calculation via born effective charge." Computer Physics Communications 307 (2025): 109425
        Modified by Jing T. on 02/20/2026 to support ase
        '''

        if self.dq is None:
            self._calculate_dq(dE)

        nmodes = self.eigenvalues.size
        nat = nmodes//3
        #hws = freqs #* 1e-3 #4.136 / 33.356
        #Raman activity
        activity = [ 0.0 for _ in range(nmodes) ]
        
        for s in range(nmodes):

            #eigvector divided by square root of mass
            eigvec = self.eigenvector[s].reshape((nat,3))
            
            #polariazation tensor
            ra_tot = np.zeros((3,3))
            for t in range(nat):
                dqt = self.dq[t]
                eigvect = eigvec[t,:]
                
                #Eq.7
                act = np.zeros((3,3,3))
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            act[i,j,k] = dqt[i,k,j]*eigvect[i]

                rat = act[0,:,:] + act[1,:,:] + act[2,:,:]
                ra_tot += rat
            
            #global constant
            ra = ra_tot /(4.0*np.pi*epslon_0)
            
            #mean polarizability derivative
            alpha = (ra[0][0] + ra[1][1] + ra[2][2])/3.0
            
            #anisotropy of the polarizability tensor derivative
            beta2 = ((ra[0][0] - ra[1][1])**2 + (ra[0][0] - ra[2][2])**2 + (ra[1][1] - ra[2][2])**2 + 6.0 * (ra[0][1]**2 + ra[0][2]**2 + ra[1][2]**2))/2.0
            
            #raman secattering activity
            activity[s] = 45.0*alpha**2 + 7.0*beta2
            #activity[s] *= 1e5
            print(f"mode {s:d} alpha = {alpha: f}, beta={beta2:f}, activity={activity[s]:f}")
        self.activity = activity

    def calculate(self, quantity = 'dq', **dq_kwargs):
        if 'quantity' == 'dq':
            self._calculate_dq()
        if 'quantity' == 'activity':
            self._calculate_dq()
    

class Phonon:
    from subprocess import check_call
    '''
    Calculator phonon eigenvector, eigenvalues using finite difference method. 
    '''
    phonon_run_name = './cache_folder/'

    def __init__(atoms, calculator, eigenvector, eigenvalues, relaxed=False):
        '''
        construct a Raman class.
        '''
        self.atoms = atoms
        self.calculator = calculator
        self.eigenvalues = None
        self.eigenvectors = None
        self.relaxed = relaxed
    
    def relax(self):
        self.atoms = self.calculator
        check_call('mkdir -p log', shell=True)
        check_call('mkdir -p traj', shell=True)
        cmd = f'rm -f ./log/{file_name}.log'
        check_call(cmd, shell=True)
        opt = LBFGS(self.atoms,
                    logfile='./log/'+file_name+'.log',
                    trajectory='./traj/'+file_name+'.traj',
                    )
        opt.run(fmax=0.01, steps=200)
        self.atoms.info['opt_converged'] = opt.converged()
        if self.atoms.info['opt_converged']:
            self.relaxed = True


    def calculate(self):
        if not self.relaxed:
            logging.info('Structure has not been relaxed explicitly. Consider trying Phonon.relax()')
        atoms.calc = calc(**calc_kwargs, **vasp_kwargs)
        N = 6
        ph = Phonons(atoms, calc(**calc_kwargs), supercell=(N, N, 1), delta=0.05, name = phonon_run_name)
        ph.run()
        # Read forces and assemble the dynamical matrix
        ph.read(acoustic=True)
        # ph.clean()
        # band_path = 'GKMG'
        # path = atoms.cell.bandpath(band_path, npoints=100)
        # bs = ph.get_band_structure(path)
        # bs.write('bs.json')

        #dos = ph.get_dos(kpts=(20, 20, 20)).sample_grid(npts=100, width=1e-3)
        #np.savetxt('dos.csv', np.column_stack([dos.get_energies(), dos.get_weights()]))    
        evals, evec = ph.band_structure([[0,0,0]], modes=True)
        self.eigenvalues = evals[0],
        self.eigenvector = np.reshape(evec, (-1,9))
