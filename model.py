import pandas as pd
import re

def extract_soc(text: str, dataset_type: str) -> str:
    """Extracts SoC name from Antutu main and AI datasets."""
    try:
        if dataset_type == 'main':
            match = re.search(r'^\d+\s+(.*?)\s*\(', str(text).replace('\n', ''))
        elif dataset_type == 'ai':
            match = re.search(r'\((.*?)\s*\d+\+\d+\)', str(text).replace('\n', ''))
        else:
            return str(text).strip()
        return match.group(1).strip() if match else str(text).strip()
    except Exception:
        return str(text)
