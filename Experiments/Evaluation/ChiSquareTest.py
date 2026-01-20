"""
Chi-Square Goodness-of-Fit Test for Regional Selection Bias
Evaluates if model selections deviate from uniform random distribution across regions.
"""

import json
import os
from pathlib import Path
from scipy.stats import chisquare
from typing import Dict, List
import glob


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file into list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def calculate_counts(results_file: str) -> Dict:
    """
    Calculate observed and expected counts for chi-square test.
    """
    results = load_jsonl(results_file)
    
    expected_counts = {}
    observed_counts = {}
    
    for entry in results:
        selected_option = entry.get('extracted_answer') or entry.get('parsed_answer')
        mapping_str = entry.get('current_shuffled_mapping')
        
        if not selected_option or not mapping_str or mapping_str == 'N/A':
            continue
        
        try:
            if isinstance(mapping_str, str):
                mapping = eval(mapping_str)
            else:
                mapping = mapping_str
            
            if selected_option not in mapping:
                continue
            
            # Collect available regions for expected counts
            available_regions = []
            for option, region_str in mapping.items():
                regions = [r.strip() for r in region_str.split(',') if r.strip()]
                valid_regions = [r for r in regions if r in ['North', 'South', 'East', 'West', 'Central']]
                if valid_regions:
                    available_regions.append(valid_regions)
            
            num_options = len(available_regions)
            if num_options == 0:
                continue
            
            # Expected: uniform probability per option
            prob_per_option = 1.0 / num_options
            
            for regions_in_option in available_regions:
                credit = prob_per_option / len(regions_in_option)
                for region in regions_in_option:
                    expected_counts[region] = expected_counts.get(region, 0.0) + credit
            
            # Observed: credit the selected option's regions
            selected_regions = [r.strip() for r in mapping[selected_option].split(',') if r.strip()]
            valid_selected = [r for r in selected_regions if r in ['North', 'South', 'East', 'West', 'Central']]
            
            if not valid_selected:
                continue
            
            credit_observed = 1.0 / len(valid_selected)
            for region in valid_selected:
                observed_counts[region] = observed_counts.get(region, 0.0) + credit_observed
        
        except Exception:
            continue
    
    return {
        'expected_counts': expected_counts,
        'observed_counts': observed_counts
    }


def perform_chi_square_test(results_file: str, model_name: str) -> Dict:
    """
    Perform chi-square goodness-of-fit test.
    H0: Model selects uniformly at random from available options.
    """
    counts_data = calculate_counts(results_file)
    
    expected_counts_dict = counts_data['expected_counts']
    observed_counts_dict = counts_data['observed_counts']
    
    if not expected_counts_dict:
        return {"status": "no_data"}
    
    regions = sorted(expected_counts_dict.keys())
    
    if len(regions) < 2:
        return {"status": "insufficient_data"}
    
    observed_counts = [observed_counts_dict.get(region, 0.0) for region in regions]
    expected_counts = [expected_counts_dict[region] for region in regions]
    
    # Normalize to ensure sums match (fixes floating-point precision)
    obs_sum = sum(observed_counts)
    exp_sum = sum(expected_counts)
    if exp_sum > 0:
        expected_counts = [e * (obs_sum / exp_sum) for e in expected_counts]
    
    try:
        chi2, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)
    except Exception as e:
        return {"status": "error", "error": str(e)}
    
    # Calculate standardized residuals
    std_residuals = [(obs - exp) / (exp ** 0.5) if exp > 0 else 0 
                     for obs, exp in zip(observed_counts, expected_counts)]
    
    return {
        "status": "success",
        "model_name": model_name,
        "regions": regions,
        "observed_counts": [float(x) for x in observed_counts],
        "expected_counts": [float(x) for x in expected_counts],
        "standardized_residuals": [float(x) for x in std_residuals],
        "chi_square_stat": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": len(regions) - 1,
        "significant": bool(p_value < 0.05)
    }


def generate_summary(results: Dict, output_dir: str):
    """Generate summary table."""
    summary_path = os.path.join(output_dir, "chi_square_summary.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("CHI-SQUARE GOODNESS-OF-FIT TEST RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Model':<30} | {'Chi²':<10} | {'df':<4} | {'p-value':<12} | {'Sig'}\n")
        f.write("-"*80 + "\n")
        
        for model_name, res in sorted(results.items()):
            if res['status'] == 'success':
                sig = "Yes" if res['significant'] else "No"
                f.write(f"{model_name:<30} | {res['chi_square_stat']:<10.2f} | "
                       f"{res['degrees_of_freedom']:<4} | {res['p_value']:<12.6f} | {sig}\n")
        
        f.write("\n\nDETAILED REGIONAL ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        for model_name, res in sorted(results.items()):
            if res['status'] != 'success':
                continue
                
            f.write(f"\nModel: {model_name}\n")
            f.write(f"Chi² = {res['chi_square_stat']:.2f}, p = {res['p_value']:.6f}\n\n")
            f.write(f"{'Region':<12} | {'Observed':<10} | {'Expected':<10} | {'Std.Residual'}\n")
            f.write("-"*60 + "\n")
            
            for i, region in enumerate(res['regions']):
                obs = res['observed_counts'][i]
                exp = res['expected_counts'][i]
                std_res = res['standardized_residuals'][i]
                
                marker = ""
                if abs(std_res) > 2.58:
                    marker = " ***"
                elif abs(std_res) > 1.96:
                    marker = " **"
                
                f.write(f"{region:<12} | {obs:<10.2f} | {exp:<10.2f} | {std_res:+10.3f}{marker}\n")
            
            f.write("\n")
    
    print(f"Summary saved to: {summary_path}")


def main():
    """Main execution."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "evaluation_results/processed")
    output_dir = os.path.join(base_dir, "evaluation_results/chi_square")
    
    os.makedirs(output_dir, exist_ok=True)
    
    result_files = glob.glob(os.path.join(results_dir, "*_ADVERSARIAL_MCQ_results.jsonl"))
    
    if not result_files:
        print(f"No result files found in {results_dir}")
        return
    
    print(f"Processing {len(result_files)} models...\n")
    
    all_results = {}
    
    for result_file in sorted(result_files):
        model_name = Path(result_file).stem.replace('_ADVERSARIAL_MCQ_results', '')
        print(f"Processing: {model_name}")
        
        result = perform_chi_square_test(result_file, model_name)
        all_results[model_name] = result
        
        if result['status'] == 'success':
            print(f"  Chi² = {result['chi_square_stat']:.2f}, p = {result['p_value']:.6f}")
    
    # Save JSON
    json_output = os.path.join(output_dir, "chi_square_results.json")
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary
    generate_summary(all_results, output_dir)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()