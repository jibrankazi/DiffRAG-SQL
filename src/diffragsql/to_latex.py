# to_latex.py
import yaml
import json
import re # <-- THIS MUST BE HERE

# Regex to strip common invalid control characters (like the backspace character ^^H)
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]") 

def clean_value(value):
    """Cleans up a value for LaTeX output."""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, str):
        return _CTRL_RE.sub('', value).replace('_', r'\_').replace('%', r'\%')
    return str(value)

def main():
    """Generates a LaTeX table from the results.json artifact."""
    
    # NOTE: Assumes the config file name is passed as an argument, 
    # but hardcoding for demo simplicity
    config_path = "configs/squad_demo.yaml"
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Use the output path defined in the config
    results_path = cfg['paths']['results_dir'] + '/results.json'

    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_path}. Run evaluate.py first.")
        return

    # Build LaTeX table structure
    latex_table = "\\begin{tabular}{l r}\n\\hline\nMetric & Value \\\\\n\\hline\n"
    
    for key, value in data.items():
        # Skip keys that are not metrics (e.g., n_test)
        if key.startswith('n_'):
            continue
        latex_table += f"{clean_value(key)} & {clean_value(value)} \\\\\n"
    
    latex_table += "\\hline\n\\end{tabular}\n"

    # Write output to paper/results.tex
    output_file = 'paper/results.tex'
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"Wrote LaTeX table to {output_file}")

if __name__ == '__main__':
    main()
