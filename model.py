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

def _extract_device_name(text: str) -> str:
    """
    Extracts the commercial device name from the Antutu AI dataset string
    """
    # Sanitizing
    try:
        text_clean = str(text).replace('\n', '')
        match = re.search(r'^\d+\s*(.*?)\(', text_clean)
        if match: # Grouping
            return match.group(1).strip()
        return text_clean.strip()
    except Exception as e:
        print(f"Warning: Device extraction failed for '{text}': {e}", file=sys.stderr)
        return str(text).strip()

def load_soc_benchmarks(filepath: str) -> pd.DataFrame:
    """
    Loads the Antutu SoC benchmark dataset to extract GPU scores
    """
    # Cleaning and processing benchmarks
    df = pd.read_csv(filepath)
    df['Normalized_SoC'] = df['Device'].apply(lambda x: _extract_soc(x, 'main'))
    df = df[['Normalized_SoC', 'GPU Score']].dropna(subset=['Normalized_SoC'])
    return df.drop_duplicates(subset=['Normalized_SoC'])

def load_ai_benchmarks(filepath: str) -> pd.DataFrame:
    """
    Loads the Antutu AI dataset to extract device-specific AI inference scores
    """
    # Cleaning and processing scores
    df = pd.read_csv(filepath)
    df['Device_Name'] = df['Device'].apply(_extract_device_name)
    df['Normalized_SoC'] = df['Device'].apply(lambda x: _extract_soc(x, 'ai'))
    df = df[['Device_Name', 'Normalized_SoC', 'Total Score']].rename(columns={'Total Score': 'Antutu_AI_Score'}).dropna()
    return df

def get_average_kernel_latency(filepath: str) -> float:
    """
    Parsing DeepEn2023 dataset to calculate the baseline latency of an AI kernel operation.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    kernel_data = data.get('conv-bn-relu', {})
    latencies = [float(metrics['latency']) for metrics in kernel_data.values() if 'latency' in metrics]
    return sum(latencies) / len(latencies)



