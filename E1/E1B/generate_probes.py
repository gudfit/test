import json
import random
import argparse

def generate_probes_from_file(input_file, output_file, num_probes=20):
    """Reads a text file of proverbs and creates fill-in-the-blank probes."""
    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    probes = []
    # Ensure we don't try to create more probes than available lines
    selected_lines = random.sample(lines, min(num_probes, len(lines)))

    for line in selected_lines:
        words = line.split()
        if len(words) > 3:  # Make sure the line is long enough to create a meaningful probe
            # Select a random word to mask, but avoid the first and last words
            mask_index = random.randint(1, len(words) - 2)
            answer = words[mask_index].strip(".,?!")
            words[mask_index] = "[MASK]"
            template = " ".join(words)
            probes.append({"template": template, "answer": answer})

    with open(output_file, 'w') as f:
        json.dump(probes, f, indent=4)
    
    print(f"Generated {len(probes)} probes and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate probes from a text file.")
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()
    generate_probes_from_file(args.input_file, args.output_file)
