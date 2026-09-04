import os
import json
import logging
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("HIMS-NVIDIA-NIM")

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "meta/llama-3.2-11b-vision-instruct"

CANCER_TYPES = [
    "breast cancer",
    "lung cancer",
    "colorectal cancer",
    "ovarian cancer",
    "pancreatic cancer",
    "prostate cancer",
    "liver cancer",
    "leukemia",
    "lymphoma"
]

AMR_IPC_CATEGORIES = [
    "amr_high",
    "amr_low",
    "amr_none",
    "ipc_adequate",
    "ipc_inadequate",
    "ipc_none"
]

class NvidiaNIMClient:
    """Client for interacting with NVIDIA NIM free hosted APIs for clinical NLP."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json"
        }

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def _call_chat_completion(self, prompt: str, system_message: str = "You are a clinical NLP assistant.") -> Optional[str]:
        if not self.is_available():
            logger.warning("NVIDIA_API_KEY is not configured. Using offline fallback.")
            return None

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 512
        }

        try:
            response = requests.post(NVIDIA_API_URL, headers=self.headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error calling NVIDIA NIM API: {e}")
            return None

    def predict_cancer_risk(self, text: str) -> Dict[str, float]:
        """Predict cancer risk probabilities using NVIDIA NIM model with offline fallback."""
        if not text or not text.strip():
            return {c: 1.0 / len(CANCER_TYPES) for c in CANCER_TYPES}

        prompt = (
            f"Analyze the following clinical note and evaluate the probability for each cancer type below.\n"
            f"Note: '{text}'\n\n"
            f"Cancer types to evaluate: {', '.join(CANCER_TYPES)}.\n"
            f"Return ONLY a valid JSON object mapping each cancer type to a float probability between 0.0 and 1.0 (probabilities should sum to 1.0).\n"
            f"Example format: {{\"{CANCER_TYPES[0]}\": 0.1, ...}}"
        )

        response_str = self._call_chat_completion(prompt)
        if response_str:
            try:
                # Extract JSON block if surrounded by markdown fence
                if "```" in response_str:
                    response_str = response_str.split("```")[1].replace("json", "").strip()
                res = json.loads(response_str)
                # Ensure all cancer types present
                probabilities = {}
                total = 0.0
                for c in CANCER_TYPES:
                    val = float(res.get(c, 0.0))
                    probabilities[c] = val
                    total += val

                if total > 0:
                    return {k: v / total for k, v in probabilities.items()}
            except Exception as e:
                logger.error(f"Failed to parse NVIDIA NIM cancer risk response: {e}")

        # --- Rule-Based Offline Fallback ---
        return self._offline_cancer_risk_fallback(text)

    def predict_amr_ipc(self, text: str) -> Dict[str, float]:
        """Predict AMR/IPC risk categories using NVIDIA NIM model with offline fallback."""
        if not text or not text.strip():
            return {cat: 1.0 / len(AMR_IPC_CATEGORIES) for cat in AMR_IPC_CATEGORIES}

        prompt = (
            f"Analyze the following clinical text for Antimicrobial Resistance (AMR) and Infection Prevention & Control (IPC) indicators:\n"
            f"Text: '{text}'\n\n"
            f"Categories: {', '.join(AMR_IPC_CATEGORIES)}.\n"
            f"Return ONLY a valid JSON object mapping each category to a probability between 0.0 and 1.0.\n"
            f"Example format: {{\"amr_high\": 0.1, \"amr_low\": 0.2, \"amr_none\": 0.7, \"ipc_adequate\": 0.8, \"ipc_inadequate\": 0.1, \"ipc_none\": 0.1}}"
        )

        response_str = self._call_chat_completion(prompt)
        if response_str:
            try:
                if "```" in response_str:
                    response_str = response_str.split("```")[1].replace("json", "").strip()
                res = json.loads(response_str)
                probabilities = {}
                for cat in AMR_IPC_CATEGORIES:
                    probabilities[cat] = float(res.get(cat, 0.0))
                return probabilities
            except Exception as e:
                logger.error(f"Failed to parse NVIDIA NIM AMR/IPC response: {e}")

        # --- Rule-Based Offline Fallback ---
        return self._offline_amr_ipc_fallback(text)

    def summarize_note(self, text: str) -> str:
        """Summarize clinical note using NVIDIA NIM LLM with offline fallback."""
        if not text or not text.strip():
            return "No content provided for summary."

        prompt = (
            f"Summarize the following clinical note concisely, highlighting key symptoms, findings, diagnoses, and management plans:\n\n"
            f"{text}\n\n"
            f"Concise Summary:"
        )

        summary = self._call_chat_completion(prompt, system_message="You are a clinical documentation assistant specializing in concise medical summaries.")
        if summary:
            return summary

        # --- Rule-Based Offline Fallback ---
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        if len(sentences) <= 2:
            return text
        return ". ".join(sentences[:2]) + "."

    def _offline_cancer_risk_fallback(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        scores = {}
        
        keywords_map = {
            "breast cancer": ["breast", "mammo", "nipple", "lump in breast", "mastectomy"],
            "lung cancer": ["lung", "hemoptysis", "coughing blood", "pulmonary nodule"],
            "colorectal cancer": ["colon", "rectal", "hematochezia", "polyp", "bowel"],
            "ovarian cancer": ["ovary", "ovarian", "pelvic mass", "adnexal"],
            "pancreatic cancer": ["pancreatic", "pancreas", "painless jaundice", "ca 19-9"],
            "prostate cancer": ["prostate", "psa", "nocturia", "prostatic"],
            "liver cancer": ["liver", "hepatic", "afp", "hepatocellular", "cirrhosis"],
            "leukemia": ["leukemia", "blast cells", "white count", "petechiae"],
            "lymphoma": ["lymphoma", "lymphadenopathy", "b symptoms", "reed-sternberg"]
        }

        matched_any = False
        for cancer, keywords in keywords_map.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                scores[cancer] = 0.5 + 0.1 * min(count, 4)
                matched_any = True
            else:
                scores[cancer] = 0.05

        if not matched_any:
            return {c: 1.0 / len(CANCER_TYPES) for c in CANCER_TYPES}

        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()}

    def _offline_amr_ipc_fallback(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        res = {
            "amr_high": 0.0, "amr_low": 0.0, "amr_none": 1.0,
            "ipc_adequate": 1.0, "ipc_inadequate": 0.0, "ipc_none": 0.0
        }

        if any(w in text_lower for w in ["mrsa", "vre", "cre", "resistant", "multidrug-resistant", "esbl"]):
            res["amr_high"] = 0.8
            res["amr_none"] = 0.1
            res["amr_low"] = 0.1
        elif "antibiotic" in text_lower or "penicillin" in text_lower:
            res["amr_low"] = 0.6
            res["amr_none"] = 0.3
            res["amr_high"] = 0.1

        if any(w in text_lower for w in ["unwashed", "no PPE", "breach", "contamination", "isolation broken"]):
            res["ipc_inadequate"] = 0.8
            res["ipc_adequate"] = 0.1
            res["ipc_none"] = 0.1
        elif any(w in text_lower for w in ["isolation", "ppe", "glove", "mask", "hand hygiene", "sanitized"]):
            res["ipc_adequate"] = 0.9

        return res
