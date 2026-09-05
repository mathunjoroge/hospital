"""
departments/pharmacy/cheminformatics.py
───────────────────────────────────────────
Deterministic Cheminformatics Engine using RDKit.
Provides SMILES validation, Lipinski Rule of 5 analysis,
3D conformation generation, and Tanimoto similarity scoring.
"""

import logging

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

logger = logging.getLogger(__name__)

# Reference Drug Library for Similarity Benchmarking
REFERENCE_DRUGS = [
    {"name": "Aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "category": "NSAID / Analgesic"},
    {"name": "Ibuprofen", "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "category": "NSAID"},
    {"name": "Paracetamol", "smiles": "CC(=O)Nc1ccc(O)cc1", "category": "Analgesic / Antipyretic"},
    {"name": "Metformin", "smiles": "CN(C)C(=N)NC(=N)N", "category": "Antidiabetic"},
    {"name": "Amoxicillin", "smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(c3ccc(cc3)O)N)C(=O)O)C", "category": "Antibiotic (Beta-lactam)"},
    {"name": "Ciprofloxacin", "smiles": "C1CC1n2cc(c(=O)c3cc(c(cc23)N4CCNCC4)F)C(=O)O", "category": "Antibiotic (Fluoroquinolone)"},
    {"name": "Atorvastatin", "smiles": "CC(C)c1c(c(c(n1CCC(CC(CC(=O)O)O)O)c2ccc(cc2)F)c3ccccc3)C(=O)Nc4ccccc4", "category": "Statin / Antihyperlipidemic"},
    {"name": "Omeprazole", "smiles": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=CC(=C3)OC", "category": "Proton Pump Inhibitor"},
    {"name": "Artemether", "smiles": "CC1CCC2C(C(C3C4(C(O3)OO2)C(CCC4C)C)OC)OC1", "category": "Antimalarial"},
    {"name": "Dexamethasone", "smiles": "CC1CC2C3CCC4=CC(=O)C=CC4(C3(C(CC2(C1(C(=O)CO)O)C)O)F)C", "category": "Corticosteroid"}
]


def validate_and_analyze_smiles(smiles: str) -> dict:
    """
    Sanitize and calculate deterministic physical descriptors for a SMILES string using RDKit.
    Returns property dictionary or error dict if invalid.
    """
    if not smiles or not isinstance(smiles, str):
        return {"is_valid": False, "error": "Empty or non-string SMILES provided."}

    clean_smiles = smiles.strip()
    mol = Chem.MolFromSmiles(clean_smiles)

    if mol is None:
        logger.warning(f"Invalid SMILES provided to RDKit validator: '{smiles}'")
        return {
            "is_valid": False,
            "smiles": clean_smiles,
            "error": "Invalid chemical SMILES syntax (RDKit parse failed)."
        }

    try:
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mw = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        hbd = int(Descriptors.NumHDonors(mol))
        hba = int(Descriptors.NumHAcceptors(mol))
        tpsa = float(Descriptors.TPSA(mol))
        rotatable_bonds = int(Descriptors.NumRotatableBonds(mol))

        # Lipinski Rule of 5 Violations
        violations = []
        if mw > 500:
            violations.append("MW > 500 Da")
        if logp > 5.0:
            violations.append("LogP > 5.0")
        if hbd > 5:
            violations.append("HBD > 5")
        if hba > 10:
            violations.append("HBA > 10")

        return {
            "is_valid": True,
            "smiles": clean_smiles,
            "canonical_smiles": canonical_smiles,
            "formula": formula,
            "mw": round(mw, 2),
            "logp": round(logp, 2),
            "hbd": hbd,
            "hba": hba,
            "tpsa": round(tpsa, 2),
            "rotatable_bonds": rotatable_bonds,
            "lipinski_pass": len(violations) == 0,
            "lipinski_violations": violations,
            "lipinski_violations_count": len(violations),
            "error": None
        }
    except Exception as e:
        logger.error(f"Error computing RDKit descriptors for '{smiles}': {e}", exc_info=True)
        return {
            "is_valid": False,
            "smiles": clean_smiles,
            "error": f"Descriptor calculation failed: {str(e)}"
        }


def generate_3d_molblock(smiles: str) -> str | None:
    """
    Generate 3D atomic coordinates and MMFF energy minimized MolBlock for 3Dmol.js rendering.
    """
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None

    try:
        mol3d = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol3d, AllChem.ETKDG())
        if res != 0:
            # Fallback to standard embedding if ETKDG fails
            res = AllChem.EmbedMolecule(mol3d, useRandomCoords=True)

        if res == 0:
            try:
                AllChem.MMFFOptimizeMolecule(mol3d, maxIters=200)
            except Exception:
                pass  # Use unoptimized 3D coords if MMFF fails
            return Chem.MolToMolBlock(mol3d)
        else:
            return Chem.MolToMolBlock(mol)
    except Exception as e:
        logger.error(f"Failed to generate 3D MolBlock for '{smiles}': {e}")
        return None


def calculate_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """
    Calculate Tanimoto similarity score (0.0 to 1.0) using Morgan Fingerprints (radius=2).
    """
    mol1 = Chem.MolFromSmiles(smiles1.strip()) if smiles1 else None
    mol2 = Chem.MolFromSmiles(smiles2.strip()) if smiles2 else None

    if not mol1 or not mol2:
        return 0.0

    try:
        fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return round(float(DataStructs.TanimotoSimilarity(fp1, fp2)), 3)
    except Exception as e:
        logger.error(f"Tanimoto calculation failed: {e}")
        return 0.0


def find_closest_reference_drugs(smiles: str, top_n: int = 3) -> list[dict]:
    """
    Find closest reference drugs in library based on Tanimoto Morgan Fingerprint similarity.
    """
    query_mol = Chem.MolFromSmiles(smiles.strip()) if smiles else None
    if not query_mol:
        return []

    try:
        query_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
        matches = []

        for ref in REFERENCE_DRUGS:
            ref_mol = Chem.MolFromSmiles(ref["smiles"])
            if ref_mol:
                ref_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
                sim = float(DataStructs.TanimotoSimilarity(query_fp, ref_fp))
                matches.append({
                    "name": ref["name"],
                    "smiles": ref["smiles"],
                    "category": ref["category"],
                    "similarity": round(sim, 3),
                    "similarity_pct": round(sim * 100, 1)
                })

        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches[:top_n]
    except Exception as e:
        logger.error(f"Reference drug similarity search failed for '{smiles}': {e}")
        return []


# Function Aliases for API compatibility
compute_molecular_properties = validate_and_analyze_smiles
find_similar_drugs = find_closest_reference_drugs
