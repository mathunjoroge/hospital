import time
import logging
import psycopg2
from typing import Dict, List, Optional, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdFingerprintGenerator, DataStructs

logger = logging.getLogger("HIMS-Cheminformatics")

DB_PARAMS = {
    'dbname': 'drugcentral',
    'user': 'drugman',
    'password': 'dosage',
    'host': 'unmtid-dbs.net',
    'port': 5433
}

# Fallback reference drugs if DB connection is unavailable
FALLBACK_DRUGS = [
    ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("Paracetamol", "CC(=O)NC1=CC=C(O)C=C1"),
    ("Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
    ("Naproxen", "CC(C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O"),
    ("Metformin", "CN(C)C(=N)N=C(N)N"),
    ("Amoxicillin", "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C"),
    ("Atorvastatin", "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4"),
    ("Omeprazole", "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=CC(=C3)OC"),
    ("Ciprofloxacin", "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O"),
    ("Metoprolol", "CC(C)NCC(COC1=CC=C(C=C1)CCOC)O"),
    ("Losartan", "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl"),
    ("Salbutamol", "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O"),
    ("Morphine", "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"),
    ("Warfarin", "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O"),
    ("Methotrexate", "CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O")
]

_CACHED_DRUGCENTRAL_FP: Optional[List[tuple]] = None

def get_drugcentral_fingerprints() -> List[tuple]:
    """Retrieve and cache Morgan fingerprints for DrugCentral approved structures."""
    global _CACHED_DRUGCENTRAL_FP
    if _CACHED_DRUGCENTRAL_FP is not None:
        return _CACHED_DRUGCENTRAL_FP

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute("SELECT name, smiles FROM structures WHERE smiles IS NOT NULL AND name IS NOT NULL LIMIT 5000")
        rows = cur.fetchall()
        conn.close()
        for name, smiles in rows:
            m = Chem.MolFromSmiles(smiles)
            if m:
                fp = gen.GetFingerprint(m)
                fps.append((name, smiles, fp))
        logger.info(f"Loaded and cached {len(fps)} structure fingerprints from DrugCentral.")
    except Exception as e:
        logger.warning(f"Could not load DrugCentral structures ({e}). Utilizing built-in fallback drug dataset.")
        for name, smiles in FALLBACK_DRUGS:
            m = Chem.MolFromSmiles(smiles)
            if m:
                fp = gen.GetFingerprint(m)
                fps.append((name, smiles, fp))
                
    _CACHED_DRUGCENTRAL_FP = fps
    return _CACHED_DRUGCENTRAL_FP

def compute_molecular_properties(smiles: str) -> Optional[Dict[str, Any]]:
    """Compute molecular properties, Lipinski rules, and 3D molblock for a given SMILES string."""
    if not smiles or not smiles.strip():
        return None
        
    mol = Chem.MolFromSmiles(smiles.strip())
    if not mol:
        return None

    # Canonical SMILES
    canonical_smiles = Chem.MolToSmiles(mol)

    # 3D conformation generation
    molblock = ""
    try:
        mol_h = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
        if res == 0:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
            molblock = Chem.MolToMolBlock(mol_h)
        else:
            # Fall back to 2D coordinates in molblock format if 3D embedding fails
            AllChem.Compute2DCoords(mol)
            molblock = Chem.MolToMolBlock(mol)
    except Exception as e:
        logger.debug(f"3D embedding warning for SMILES {smiles}: {e}")
        AllChem.Compute2DCoords(mol)
        molblock = Chem.MolToMolBlock(mol)

    mw = round(Descriptors.MolWt(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 2)
    tpsa = round(Descriptors.TPSA(mol), 2)
    hbd = int(Descriptors.NumHDonors(mol))
    hba = int(Descriptors.NumHAcceptors(mol))
    rotb = int(Descriptors.NumRotatableBonds(mol))

    # Lipinski Rule of 5 check
    violations = []
    if mw > 500:
        violations.append(f"MW {mw} > 500")
    if logp > 5:
        violations.append(f"LogP {logp} > 5")
    if hbd > 5:
        violations.append(f"HBD {hbd} > 5")
    if hba > 10:
        violations.append(f"HBA {hba} > 10")

    return {
        "smiles": canonical_smiles,
        "raw_smiles": smiles,
        "mw": mw,
        "logp": logp,
        "tpsa": tpsa,
        "hbd": hbd,
        "hba": hba,
        "rotb": rotb,
        "lipinski_violations_count": len(violations),
        "lipinski_violations": violations,
        "lipinski_pass": len(violations) <= 1,
        "molblock": molblock
    }

def find_similar_drugs(smiles: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """Perform Tanimoto similarity search against cached DrugCentral structures."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return []

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    query_fp = gen.GetFingerprint(mol)
    db_fps = get_drugcentral_fingerprints()

    results = []
    for name, d_smiles, fp in db_fps:
        sim = DataStructs.TanimotoSimilarity(query_fp, fp)
        results.append({
            "name": name,
            "smiles": d_smiles,
            "similarity": round(sim * 100, 1)
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_n]
