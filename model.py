import pandas as pd
import re
import json
import os
import sys

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


def calculate_integration_capacity(df: pd.DataFrame, base_kernel_latency: float) -> pd.DataFrame:
    """
    Modeling the threshold of concurrent AI operations during 3D rendering.
    Assuming 60 FPS budget
    """
    FRAME_BUDGET_MS = 16.66
    df = df[(df['Antutu_AI_Score'] > 0) & (df['GPU Score'] > 0)].copy()
    median_ai_score = df['Antutu_AI_Score'].median()
    max_gpu_score = df['GPU Score'].max()

    # Scaling
    df['Device_AI_Kernel_Latency_ms'] = base_kernel_latency * (median_ai_score / df['Antutu_AI_Score'])

    # Calculate MAX operations (no 3d)
    df['Absolute_Max_AI_Ops'] = FRAME_BUDGET_MS / df['Device_AI_Kernel_Latency_ms']

    # GPU ranking
    df['GPU_Percentile'] = df['GPU Score'] / max_gpu_score

    # ops allowed without 3d rendering taking effect on performance
    df['Theoretical_Max_AI_Ops_Per_Frame'] = (df['Absolute_Max_AI_Ops'] * df['GPU_Percentile']).astype(int)

    output_cols = [
        'Device_Name', 'Normalized_SoC', 'GPU Score', 'Antutu_AI_Score',
        'Device_AI_Kernel_Latency_ms', 'GPU_Percentile', 'Theoretical_Max_AI_Ops_Per_Frame'
    ]
    return df[output_cols].sort_values(by='Theoretical_Max_AI_Ops_Per_Frame', ascending=False)


def main():
    """
    Main driver function
    """

    antutu_soc_path = '../Antutu/Android_SoC.csv'
    antutu_ai_path = '../Antutu/Android_AI_General.csv'
    deepen_latency_path = '../DeepEn/Kernel_latency/conv-bn-relu_latency.json'

    # Ingestion
    df_soc = load_soc_benchmarks(antutu_soc_path)
    df_ai = load_ai_benchmarks(antutu_ai_path)
    avg_kernel_latency = get_average_kernel_latency(deepen_latency_path)

    # Left joining
    merged_df = pd.merge(df_ai, df_soc, on='Normalized_SoC', how='left')

    # Calculations
    final_df = calculate_integration_capacity(merged_df, avg_kernel_latency)

    # Outputting
    output_dir = '.'
    output_file = f'{output_dir}/merged_metrics.csv'

    try:
        # Ensuring output dir exists
        os.makedirs(output_dir, exist_ok=True)
        final_df.to_csv(output_file, index=False)
        print(f"Data successfully merged and exported to: {output_file}")
    except Exception as e:
        print(f"Failed to write output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
