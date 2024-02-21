import argparse
import json
import logging
import os
import pandas as pd
import numpy as np
from Bio import SeqIO
from keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_fasta_sequences(fasta_file):
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

def load_configuration(config_file):
    with open(config_file) as f:
        config = json.load(f)
    return config

def load_annotations(file_path,):
    annotations = pd.read_csv(file_path, sep='\t')
    return annotations

def encode_sequences(sequences, max_length):
    # Dummy encoding function - replace with actual sequence encoding logic
    # For simplicity, this example assumes sequences are converted to a fixed-size vector.
    encoded = {k: np.random.rand(max_length) for k, _ in sequences.items()}
    return encoded

def predict_go_terms(models, encoded_sequences, train_annotations):
    predictions = {}
    counter = 0
    for model in models.values():
        X = np.array(list(encoded_sequences.values()))
        y_pred = model.predict(X)
        predictions[train_annotations["GO_term"][counter]] = y_pred
        counter += 1
    return predictions

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta_file', help='Path to the FASTA file containing protein sequences.')
    parser.add_argument('-conf_file', help='Configuration file path', required=True)
    parser.add_argument('-out_dir', help='Output directory', default='.')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s]  %(message)s')

    # Load the configuration file
    config = load_configuration(args.conf_file)

    # Loading the Go_term
    train_annotations = load_annotations(config['annotations_file'])

    # Load sequences from the FASTA file
    sequences = load_fasta_sequences(args.fasta_file)
    logging.info(f"Loaded {len(sequences)} sequences from {args.fasta_file}")

    # Encode sequences
    max_length = 1024  # Example max length, adjust based on your model requirements
    encoded_sequences = encode_sequences(sequences, max_length)
    logging.info(f"Encoded sequences into fixed-length vectors")

    # Load models
    models = {
        'cellular_component': load_model(config['cellular_component_model']),
        'molecular_function': load_model(config['molecular_function_model']),
        'biological_process': load_model(config['biological_process_model']),
    }

    # Predict GO terms
    predictions = predict_go_terms(models, encoded_sequences, train_annotations)
    logging.info("Predicted GO terms for the sequences")

    # Example of processing predictions into a DataFrame
    # Adjust according to the structure of your predictions
    protein_ids = list(encoded_sequences.keys())
    dfs = []
    for aspect, preds in predictions.items():
        for i, protein_id in enumerate(protein_ids):
            for go_term, prob in enumerate(preds[i]):
                dfs.append(pd.DataFrame({'Protein ID': protein_id, 'GO Term': f"{aspect}_{go_term}", 'Probability': prob}, index=[0]))
    result_df = pd.concat(dfs, ignore_index=True)
    result_df = result_df.sort_values(ascending=False, by="Probability")

    # Save the DataFrame to a file
    output_file = os.path.join(args.out_dir, "predictions.tsv")
    result_df.to_csv(output_file, sep='\t', index=False)
    logging.info(f"Saved predictions to {output_file}")

if __name__ == "__main__":
    main()
