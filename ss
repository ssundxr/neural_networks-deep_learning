embed_dim = 128
num_heads = 4

inputs = layers.Input(shape=(max_len,))

x = PositionalEmbedding(max_len, max_features, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, num_heads)(x)

# Convert token embeddings → sentence embedding
sentence_embedding = layers.GlobalAveragePooling1D()(x)

embedding_model = keras.Model(
    inputs=inputs,
    outputs=sentence_embedding
)

embedding_model.summary()
