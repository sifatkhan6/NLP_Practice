from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Sample dataset (list of sentences with human names, car names, and place names)
sample_data = [
    "John drives a Tesla to New York.",
    "Alice owns a BMW and lives in Paris.",
    "Michael rented a Ford in Los Angeles.",
    "Sarah bought a Mercedes in Berlin.",
    "David prefers to drive an Audi in Tokyo.",
    "Jessica has a Honda and loves visiting London.",
    "Daniel's favorite car is a Toyota and he works in Sydney.",
    "Emily's Chevrolet is parked in San Francisco.",
    "Matthew loves his Porsche and frequently travels to Rome.",
    "Sophia just purchased a Nissan and plans to drive to Barcelona."
]

# Preprocess the sample data
preprocessed_data = [simple_preprocess(sentence) for sentence in sample_data]

# Train a Word2Vec model
model = Word2Vec(
    sentences=preprocessed_data,
    vector_size=100,
    window=4,
    min_count=1,
    workers=4)

model.build_vocab(preprocessed_data, progress_per=100)

model.epochs
print("Model Epochs is : ", model.epochs)

model.corpus_count
print("Model Corpus is : ", model.corpus_count)

model.train(preprocessed_data, total_examples=model.corpus_count, epochs=model.epochs)

sm = model.wv.most_similar('emily', topn=5)
for data in sm:
    print(data)

print(model.wv.similarity(w1="loves", w2="london"))


# Example: Get the vector for a specific word
# word_vector = model.wv['toyota']
# print(f"Vector for 'Tesla':\n{word_vector}")

# Example: Find similar words
# similar_words = model.wv.most_similar('toyota', topn=5)
# print(f"Words most similar to 'Tesla':\n{similar_words}")

# Example: Find similar words to a place name
# similar_places = model.wv.most_similar('paris', topn=5)
# print(f"Words most similar to 'Paris':\n{similar_places}")