import psycopg2
from psycopg2.extras import RealDictCursor
from pprint import pprint
import logging

# Database credentials
db_params = {
    'dbname': 'drugcentral',
    'user': 'drugman',
    'password': 'dosage',
    'host': 'unmtid-dbs.net',
    'port': '5433'
}

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Function to get database connection
def get_db_connection():
    return psycopg2.connect(**db_params)

# Test the query
def test_query(struct_id):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
SELECT 
    s.id AS struct_id,
    s.cd_id,
    s.name,
    s.cas_reg_no,
    s.cd_formula,
    s.cd_molweight,
    s.inchikey,
    s.smiles,
    s.status,
    s.stem,

    -- Active Ingredients
    COALESCE(JSONB_AGG(DISTINCT JSONB_BUILD_OBJECT(
        'id', ai.id,
        'ndc_product_code', ai.ndc_product_code,
        'substance_name', ai.substance_name,
        'substance_unii', ai.substance_unii,
        'active_moiety_name', ai.active_moiety_name,
        'active_moiety_unii', ai.active_moiety_unii,
        'quantity', ai.quantity,
        'unit', ai.unit
    )) FILTER (WHERE ai.id IS NOT NULL), '[]'::JSONB) AS active_ingredients,

    -- Structure Type
    COALESCE(JSONB_AGG(DISTINCT JSONB_BUILD_OBJECT(
        'type', st.type
    )) FILTER (WHERE st.type IS NOT NULL), '[]'::JSONB) AS structure_types,

    -- Pharma Classes
    COALESCE(JSONB_AGG(DISTINCT JSONB_BUILD_OBJECT(
        'class_code', pc.class_code,
        'name', pc.name,
        'type', pc.type,
        'source', pc.source
    )) FILTER (WHERE pc.class_code IS NOT NULL), '[]'::JSONB) AS pharma_classes,

    -- PDB Entries
    COALESCE(JSONB_AGG(DISTINCT JSONB_BUILD_OBJECT(
        'pdb_id', pdb.pdb,
        'title', pdb.title,
        'exp_method', pdb.exp_method,
        'ligand_id', pdb.ligand_id,
        'chain_id', pdb.chain_id,
        'deposition_date', pdb.deposition_date
    )) FILTER (WHERE pdb.pdb IS NOT NULL), '[]'::JSONB) AS pdb_entries,

    -- ActTableFull
    COALESCE(JSONB_AGG(DISTINCT JSONB_BUILD_OBJECT(
        'target_id', act.target_id,
        'action_type', act.action_type,
        'act_value', act.act_value,
        'act_unit', act.act_unit,
        'act_type', act.act_type,
        'target_name', act.target_name,
        'target_class', act.target_class,
        'organism', act.organism,
        'gene', act.gene,
        'swissprot', act.swissprot
    )) FILTER (WHERE act.target_id IS NOT NULL), '[]'::JSONB) AS act_table_full,

    -- OMOP Relationship
    COALESCE(JSONB_AGG(DISTINCT JSONB_BUILD_OBJECT(
        'omop_id', omop.concept_id,
        'concept_name', omop.concept_name,
        'snomed_conceptid', omop.snomed_conceptid,
        'umls_cui', omop.umls_cui,
        'snomed_full_name', omop.snomed_full_name,
        'cui_semantic_type', omop.cui_semantic_type,
        'relationship_name', omop.relationship_name
    )) FILTER (WHERE omop.concept_id IS NOT NULL), '[]'::JSONB) AS omop_relationships,

    -- FAERS Entries
    COALESCE(JSONB_AGG(DISTINCT JSONB_BUILD_OBJECT(
        'faers_id', faers.id,
        'meddra_code', faers.meddra_code,
        'meddra_name', faers.meddra_name,
        'llr', faers.llr,
        'llr_threshold', faers.llr_threshold,
        'drug_ae', faers.drug_ae,
        'drug_no_ae', faers.drug_no_ae,
        'no_drug_ae', faers.no_drug_ae,
        'no_drug_no_ae', faers.no_drug_no_ae,
        'level', faers.level
    )) FILTER (WHERE faers.id IS NOT NULL), '[]'::JSONB) AS faers_entries

FROM structures s
LEFT JOIN active_ingredient ai ON s.id = ai.struct_id
LEFT JOIN structure_type st ON s.id = st.struct_id
LEFT JOIN pharma_class pc ON s.id = pc.struct_id
LEFT JOIN pdb ON s.id = pdb.struct_id
LEFT JOIN act_table_full act ON s.id = act.struct_id
LEFT JOIN omop_relationship omop ON s.id = omop.struct_id
LEFT JOIN faers ON s.id = faers.struct_id
WHERE s.id = %s
GROUP BY s.id, s.cd_id
            """

            cur.execute(query, (struct_id,))
            drug = cur.fetchone()

            if not drug:
                print(f"No drug found with struct_id = {struct_id}")
            else:
                print("Drug details:")
                pprint(drug)  # Pretty-print the output

    except psycopg2.Error as e:
        logging.error(f"Database error: {e.pgerror}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        if conn:
            conn.close()

# Run the test
if __name__ == "__main__":
    struct_id = 1444  # Replace with the struct_id you want to test
    test_query(struct_id)