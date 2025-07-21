import time
import lmdb
import pyscf
from tqdm import tqdm
from pyscf import lib, scf
import numpy as np
import json 
import os
np.random.seed(123)

def compute_dm(mo_occ, mo_coeff):
    mocc = mo_coeff[:,mo_occ>0]
    dm = (mocc*mo_occ[mo_occ>0]).dot(mocc.conj().T)
    return dm

def work_lmdb(file_path: str, file_name: str):
    lmdb_env = lmdb.open(file_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    db = lmdb.open(
        file_path+file_name,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
  
    dataset = LmdbDataset({"src": file_path+file_name})
    return dataset

def main():
    lib.num_threads(24)
    print("N THREADS:", lib.num_threads())
    
    filename = None#"/home/ihor/projects/MLGuessQC/datasets/tmQM_dft_acceleration_estimate_npz"
    output_dir = "pyscf_experiments/results_debug"
    
    if filename == None:
        mol = pyscf.gto.Mole()
        mol.atom = pyscf.gto.mole.fromfile('/home/ihor/projects/MLGuessQC/datasets/JAXREF.xyz')
        mol.spin = 1
        mol.basis = 'LANL2DZ'
        mol.verbose = 4
        # mol.build(
        #     atom = '''C 0 0 0; O 0 1 0''',
        #     basis = 'sto-3g')
        mf = mol.KS()
        mf.max_cycle = 200
        mf.init_guess = 'vsap'
        mf = scf.addons.smearing_(mf, sigma=0.01, method='fermi')
        mf.xc = 'R2SCAN'
        start = time.time()
        mf.kernel()
        end = time.time()
        calc_time = end - start
        print("Energy:", mf.e_tot)
        print("Total calculation time:", calc_time)
        veff = mf.get_veff()
        dm = mf.dm
        import matplotlib.pyplot as plt

        # Plot the density matrix as an image
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(dm, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Density Matrix Values')
        plt.title('Density Matrix')
        plt.xlabel('Index')
        plt.ylabel('Index')
        plt.savefig("pyscf_experiments/temp/dm_FEQCOT_def2svp.png")
        plt.show()
        
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(veff, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='MO Coeff Matrix Values')
        plt.title('MO Coeff Matrix')
        plt.xlabel('Index')
        plt.ylabel('Index')
        plt.savefig("pyscf_experiments/temp/veff_FEQCOT_def2svp.png")
        plt.show()
        
        # dm_scf = compute_dm(mf.mo_occ, mf.mo_coeff)
        # dm_guess = compute_dm(np.array([2., 2., 2., 2., 2., 2., 2., 0., 0., 0.]), mf.mo_coeff)
        # assert np.allclose(dm_scf, dm_guess), "Density matrices do not match!"
        
    elif os.path.isdir(filename) and any(".npz" in f for f in os.listdir(filename)):
        npz_files = [f for f in os.listdir(filename) if ".npz" in f]
        for npz_file in npz_files:
            try:
                file_path = os.path.join(filename, npz_file)
                data = np.load(file_path)
                
                numbers = data["atomic_numbers"]
                positions = data["positions"]
                
                atom_symbols = [pyscf.gto.mole._symbol(int(num)) for num in numbers]
                atom_positions = ["{:.8f} {:.8f} {:.8f}".format(*pos) for pos in positions]
                atom_strings = [f"{symbol} {pos}" for symbol, pos in zip(atom_symbols, atom_positions)]
                mol_string = "\n".join(atom_strings)
                mol = pyscf.gto.Mole()
                mol.atom = pyscf.gto.mole.fromstring(mol_string, format='raw')
                
                mol.basis = 'LANL2DZ'
                mol.verbose = 4
                mol.spin = 0
                mf = mol.KS()
                mf.init_guess = 'vsap'
                mf = scf.addons.smearing_(mf, sigma=0.01, method='fermi')
                mf.xc = 'R2SCAN'
                mf.max_cycle = 300
                
                if 'predicted_dm' in data:
                    dm0 = data['predicted_dm']
                    print(f"!!! Using predicted DM from {npz_file} !!!")
                else:
                    dm0 = None
                
                start = time.time()
                mf.kernel(dm0=dm0)
                end = time.time()
                calc_time = end - start
                print(f"Total calculation time: {calc_time/60:.2f} minutes")
                    
                n_scf_iter = np.array([mf.cycles])
                init_dm = mf.dm_init
                final_dm = mf.dm
                final_energy = np.array([mf.e_tot])
                calc_time = np.array([calc_time])
                h1e = mf.get_hcore()
                s1e = mf.get_ovlp()
                fock = mf.get_fock()
                veff = mf.get_veff()
                
                save_path = os.path.join(output_dir, npz_file.replace(".npz", "_pyscf_result.npz"))
                np.savez(save_path,
                    atomic_numbers=numbers,
                    positions=positions,
                    init_dm=init_dm,
                    final_dm=final_dm, 
                    h1e=h1e,
                    s1e=s1e,
                    final_fock=fock,
                    final_veff=veff,
                    final_energy=final_energy,
                    n_scf_iter=n_scf_iter,
                    calc_time=calc_time,
                )
            except Exception as e:
                print(f"Failed to process {npz_file}: {e}")
                continue
 
  
    elif filename.endswith(".xyz"):
        mol = pyscf.gto.Mole()
        mol.atom = pyscf.gto.mole.fromfile(filename)
        # mol.basis = '631g'
        mol.basis = 'def2tzvpd'
        mol.verbose = 0
        mf = mol.KS()
        # mf.xc = 'b3lyp'
        mf.xc = 'wb97m_v'
        mf.kernel()
        
        n_scf_iter = np.array([mf.cycles])
        calc_time = np.array([calc_time])
        init_dm = mf.dm_init
        init_mo_energy = mf.mo_energy_init
        init_mo_coeff = mf.mo_coeff_init
        final_dm = mf.dm
        final_mo_energy = mf.mo_energy
        final_mo_coeff = mf.mo_coeff
        final_energy = np.array([mf.e_tot])
    
    elif filename.endswith(".lmdb"):
        dataset = work_lmdb("/home/ihor/projects/MLGuessQC/data/quinones/", filename)
        # Load the structure from the LMDB file
        failed_idx = []
        calculated_idx = []
        for i in tqdm(range(0,200)):
            if len(calculated_idx) == 100:
                break
            try:
                molecule = dataset[i]
                numbers = molecule["atomic_numbers"].detach().numpy()
                positions = molecule["pos"].detach().numpy()
                
                # Convert atomic numbers and positions to raw string format
                atom_symbols = [pyscf.gto.mole._symbol(int(num)) for num in numbers]
                atom_positions = ["{:.8f} {:.8f} {:.8f}".format(*pos) for pos in positions]
                atom_strings = [f"{symbol} {pos}" for symbol, pos in zip(atom_symbols, atom_positions)]
                mol_string = "\n".join(atom_strings)
                mol = pyscf.gto.Mole()
                mol.atom = pyscf.gto.mole.fromstring(mol_string, format='raw')
                
                mol.basis = '631g'
                #mol.basis = 'def2svpd'
                mol.verbose = 0
                mf = mol.KS()
                mf.xc = 'b3lyp'
                #mf.xc = 'wB97M-V'
                #mf.nlc='VV10'
                mf.max_cycle = 100
                
                final_dm = np.load(f"pyscf_experiments/results/{molecule.cid}_pyscf_result.npz")["final_dm"]
                dm0 = final_dm + np.random.normal(0, 0.01, final_dm.shape)
                
                # mo_energy, mo_coeff = [], []
                # mo_occ = mf.get_occ(mo_energy, mo_coeff)
                # dm0 = mf.make_rdm1(mo_coeff, mo_occ)
                
                
                start = time.time()
                mf.kernel(dm0)
                end = time.time()
                calc_time = end - start
                
                with open("pyscf_experiments/temp/ddm.json", "r") as file:
                    data = json.load(file)
                    ddm = np.array(data["ddm_values"])
                os.remove("pyscf_experiments/temp/ddm.json")
                
                n_scf_iter = np.array([mf.cycles])
                calc_time = np.array([calc_time])
                init_dm = mf.dm_init
                init_mo_energy = mf.mo_energy_init
                init_mo_coeff = mf.mo_coeff_init
                final_dm = mf.dm
                final_mo_energy = mf.mo_energy
                final_mo_coeff = mf.mo_coeff
                final_energy = np.array([mf.e_tot])
                
                np.savez(f"pyscf_experiments/results_noisy_0p01std/{molecule.cid}_pyscf_result", 
                        init_dm=init_dm, 
                        init_mo_energy=init_mo_energy,
                        init_mo_coeff=init_mo_coeff,
                        final_dm=final_dm, 
                        final_mo_energy=final_mo_energy, 
                        final_mo_coeff=final_mo_coeff,
                        final_energy=final_energy,
                        n_scf_iter=n_scf_iter,
                        calc_time=calc_time,
                        ddm=ddm)
                
                calculated_idx.append(i)
            
            except Exception as e:
                print(f"Failed to calculate for index {i}")
                failed_idx.append(i)
                continue
            
        print("Failed indices:", failed_idx)
        # Save the failed indices to a file
        with open("failed_indices.json", "w") as f:
            json.dump(failed_idx, f)
        
        
            
            

    # Orbital energies, Mulliken population etc.
    #mf.analyze()

if __name__ == "__main__":
    main()