import pandas as pd
import re

def _extract_soc(text: str, dataset_type: str) -> str:
    """
    Extracts and normalizes the SoC name from Anuntu datasets"
    """
    # Sanitizing input
    try:
        text_clean = str(text).replace('\n', '')
        if dataset_type == 'main': # main database
            match = re.search(r'^\d+\s+(.*?)\s*\(', text_clean)
            extracted = match.group(1) if match else text_clean
        elif dataset_type == 'ai': # ai database
            match = re.search(r'\((.*?)\)', text_clean)
            extracted = match.group(1) if match else text_clean
            extracted = re.sub(r'\s*\d+\+\d+.*$', '', extracted)
        else:
            extracted = text_clean
        # Normalization
        extracted = extracted.lower().strip()
        extracted = re.sub(r'\bs-', 'snapdragon ', extracted)
        extracted = re.sub(r'\bd-', 'dimensity ', extracted)
        extracted = re.sub(r'\b(qualcomm|mediatek|apple|samsung|hisilicon)\s+', '', extracted)
        return extracted.strip()
    except Exception as e:
        print(f"Warning: SoC extraction failed for '{text}': {e}", file=sys.stderr)
        return str(text).strip()

