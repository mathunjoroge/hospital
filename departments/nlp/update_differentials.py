from departments.nlp.batch_processing import query_uts_api
from departments.nlp.logging_setup import logger
import json
import os
from dotenv import load_dotenv

load_dotenv()
UTS_API_KEY = os.getenv("UTS_API_KEY", "mock_api_key")
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "/home/mathu/projects/hospital/departments/nlp/knowledge_base")
CLINICAL_PATHWAYS_PATH = os.path.join(KNOWLEDGE_BASE_DIR, "clinical_pathways.json")

def update_obesity_differentials():
    try:
        # Query UMLS for obesity
        result = query_uts_api("obesity", api_key=UTS_API_KEY)
        related_conditions = result.get("related_conditions", [])
        if not related_conditions:
            logger.warning("No related conditions found for obesity. Using defaults.")
            differentials = ["Type 2 Diabetes", "Metabolic Syndrome"]
        else:
            differentials = [condition["name"] for condition in related_conditions[:2]]  # Take at least 2
            logger.info(f"Retrieved differentials for obesity: {differentials}")

        # Load clinical_pathways.json
        with open(CLINICAL_PATHWAYS_PATH, 'r') as f:
            clinical_pathways = json.load(f)

        # Update musculoskeletal.obesity.differentials
        if "musculoskeletal" in clinical_pathways and "obesity" in clinical_pathways["musculoskeletal"]:
            clinical_pathways["musculoskeletal"]["obesity"]["differentials"] = differentials
            logger.info("Updated differentials for musculoskeletal.obesity")
        else:
            logger.warning("musculoskeletal.obesity not found in clinical_pathways.json")
            return

        # Save updated file
        with open(CLINICAL_PATHWAYS_PATH, 'w') as f:
            json.dump(clinical_pathways, f, indent=2)
        logger.info(f"Updated {CLINICAL_PATHWAYS_PATH}")
    except Exception as e:
        logger.error(f"Failed to update clinical_pathways.json: {str(e)}")

if __name__ == "__main__":
    update_obesity_differentials()