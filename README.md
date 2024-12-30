# Skip-Gram Model from Scratch

This repository demonstrates the implementation of the Skip-Gram model from scratch using Python and NumPy. The project explores word embeddings and their applications in natural language processing tasks. Key functionalities include training the model on Vietnamese text data, visualizing word embeddings, and analyzing cosine similarities between word pairs.

## Features

- Build a word embedding model from scratch without relying on external libraries like TensorFlow or PyTorch.
- Train on Vietnamese text data with adjustable hyperparameters such as embedding size, learning rate, and batch size.
- Use PCA to reduce the embedding dimensions and visualize the word vectors.
- Analyze cosine similarity between words to understand semantic relationships.

## Dependencies

- Python 3.x
- Pyvi
- NumPy
- Matplotlib
- NLTK (for text preprocessing)

## Results

- Successfully trained a Skip-Gram model to generate meaningful word embeddings for Vietnamese language.
- Visualized word embeddings using PCA, demonstrating clustering of semantically similar words.
- Observed high cosine similarity between synonymous word pairs and low or negative similarity between unrelated words.

## Future Improvements

- Implement advanced preprocessing techniques, including stemming or lemmatization.
- Use larger datasets to improve embedding quality.
- Experiment with other dimensionality reduction methods like t-SNE or UMAP for visualization.

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code with proper attribution. See the LICENSE file for details.

## Contributing
This repository is intended as a reference for learning purposes. Feel free to explore the code and share feedback if you want :)
