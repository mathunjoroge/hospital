import psycopg2
from psycopg2.extras import RealDictCursor

# Update these with your actual DB credentials
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432
POSTGRES_DB = "hospital_umls"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "postgres"

default_symptoms = {
    "respiratory": {
        "facial pain": {"description": "UMLS-derived: facial pain", "umls_cui": "C0234450", "semantic_type": "Sign or Symptom"},
        "nasal congestion": {"description": "UMLS-derived: nasal congestion", "umls_cui": "C0027424", "semantic_type": "Sign or Symptom"},
        "purulent nasal discharge": {"description": "UMLS-derived: purulent nasal discharge", "umls_cui": "C0242209", "semantic_type": "Sign or Symptom"},
        "cough": {"description": "UMLS-derived: cough", "umls_cui": "C0010200", "semantic_type": "Sign or Symptom"}
    },
    "infectious": {
        "fever": {"description": "UMLS-derived: fever", "umls_cui": "C0015967", "semantic_type": "Sign or Symptom"}
    },
    "neurological": {
        "headache": {"description": "UMLS-derived: headache", "umls_cui": "C0018681", "semantic_type": "Sign or Symptom"},
        "photophobia": {"description": "UMLS-derived: photophobia", "umls_cui": "C0085636", "semantic_type": "Sign or Symptom"},
        "neck stiffness": {"description": "UMLS-derived: neck stiffness", "umls_cui": "C0029101", "semantic_type": "Sign or Symptom"}
    },
    "cardiovascular": {
        "chest pain": {"description": "UMLS-derived: chest pain", "umls_cui": "C0008031", "semantic_type": "Sign or Symptom"},
        "shortness of breath": {"description": "UMLS-derived: shortness of breath", "umls_cui": "C0013144", "semantic_type": "Sign or Symptom"},
        "chest tightness": {"description": "UMLS-derived: chest tightness", "umls_cui": "C0242209", "semantic_type": "Sign or Symptom"}
    },
    "dermatological": {
        "rash": {"description": "UMLS-derived: rash", "umls_cui": "C0015230", "semantic_type": "Sign or Symptom"}
    },
    "musculoskeletal": {
        "back pain": {"description": "UMLS-derived: back pain", "umls_cui": "C0004604", "semantic_type": "Sign or Symptom"},
        "knee pain": {"description": "UMLS-derived: knee pain", "umls_cui": "C0231749", "semantic_type": "Sign or Symptom"},
        "joint pain": {"description": "UMLS-derived: joint pain", "umls_cui": "C0003862", "semantic_type": "Sign or Symptom"},
        "pain on movement": {"description": "UMLS-derived: pain on movement", "umls_cui": "C0234452", "semantic_type": "Sign or Symptom"}
    },
    "gastrointestinal": {
        "epigastric pain": {"description": "UMLS-derived: epigastric pain", "umls_cui": "C0234451", "semantic_type": "Sign or Symptom"},
        "nausea": {"description": "UMLS-derived: nausea", "umls_cui": "C0027497", "semantic_type": "Sign or Symptom"}
    },
    "general": {
        "fatigue": {"description": "UMLS-derived: fatigue", "umls_cui": "C0013144", "semantic_type": "Sign or Symptom"},
        "obesity": {"description": "UMLS-derived: obesity", "umls_cui": "C0028754", "semantic_type": "Disease or Syndrome"}
    }
}

def lookup_umls(term, conn):
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("""
            SELECT c.cui, c.str, sty.sty
            FROM umls.MRCONSO c
            JOIN umls.MRSTY sty ON c.cui = sty.cui
            WHERE LOWER(c.str) = %s AND c.sab = 'SNOMEDCT_US' AND c.suppress = 'N'
            LIMIT 1
        """, (term.lower(),))
        result = cursor.fetchone()
        if result:
            return result['cui'], result['sty']
        # Try LIKE if exact match fails
        cursor.execute("""
            SELECT c.cui, c.str, sty.sty
            FROM umls.MRCONSO c
            JOIN umls.MRSTY sty ON c.cui = sty.cui
            WHERE LOWER(c.str) LIKE %s AND c.sab = 'SNOMEDCT_US' AND c.suppress = 'N'
            LIMIT 1
        """, ('%' + term.lower() + '%',))
        result = cursor.fetchone()
        if result:
            return result['cui'], result['sty']
        return None, None

def update_symptoms(default_symptoms, conn):
    updated = {}
    for category, symptoms in default_symptoms.items():
        updated[category] = {}
        for symptom, info in symptoms.items():
            cui, sty = lookup_umls(symptom, conn)
            if cui and sty:
                print(f"Updating '{symptom}': CUI={cui}, SemanticType={sty}")
                info['umls_cui'] = cui
                info['semantic_type'] = sty
            else:
                print(f"WARNING: No UMLS match for '{symptom}'")
            updated[category][symptom] = info
    return updated

def main():
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )
    try:
        updated = update_symptoms(default_symptoms, conn)
        print("\n# --- Updated default_symptoms --- #\n")
        import pprint
        pprint.pprint(updated)
    finally:
        conn.close()

if __name__ == "__main__":
    main()