# E2/tt/src/data_handler.py
import re
from datasets import load_dataset

def get_sentences_from_dataset(config: dict) -> list[str]:
    print(f"Loading dataset: {config['dataset_name']}, subset: {config['dataset_subset']}...")
    dataset = load_dataset(config['dataset_name'], config['dataset_subset'])
    text_data = dataset[config['test_split']]['text']

    all_sentences = []
    for paragraph in text_data:
        sentences = paragraph.split('\n')
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if clean_sentence and not re.match(r'^=.*=$', clean_sentence):
                all_sentences.append(clean_sentence)
    
    print(f"Found {len(all_sentences)} sentences in the {config['test_split']} split.")
    return all_sentences

if __name__ == "__main__":
    print("--- Running data_handler.py self-test ---")
    mock_config = {
        'dataset_name': 'wikitext',
        'dataset_subset': 'wikitext-2-raw-v1',
        'test_split': 'validation'
    }
    sentences = get_sentences_from_dataset(mock_config)
    
    print(f"\nSuccessfully loaded {len(sentences)} sentences.")
    print("First 5 sentences:")
    for s in sentences[:5]:
        print(f"- {s}")
