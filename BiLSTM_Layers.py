from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout, Multiply, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf

def define_model(MAX_LEN, vocab):
    # Input layers
    premise_input = Input(shape=(MAX_LEN,), dtype='int32', name='premise_input')
    hypothesis_input = Input(shape=(MAX_LEN,), dtype='int32', name='hypothesis_input')

    # Embedding layer
    embedding_dim = 100  # You can adjust this
    vocab_size = len(vocab) + 1  # Adding 1 for padding index

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)

    premise_embedded = embedding_layer(premise_input)
    hypothesis_embedded = embedding_layer(hypothesis_input)

    # BiLSTM layer
    bi_lstm = Bidirectional(LSTM(64, return_sequences=True))  

    premise_lstm = bi_lstm(premise_embedded)
    hypothesis_lstm = bi_lstm(hypothesis_embedded)
    
    premise_attention = attention_layer(premise_lstm)
    hypothesis_attention = attention_layer(hypothesis_lstm)

    # Combine premise & hypothesis attention outputs
    combined = Multiply()([premise_attention, hypothesis_attention])  

    # Fully connected layers
    dense = Dense(64, activation='relu')(combined)
    dropout = Dropout(0.5)(dense)
    output = Dense(3, activation='softmax')(dropout)
    
    # Define and compile model
    model = Model(inputs=[premise_input, hypothesis_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Summary
    model.summary()

    return model

# Attention Mechanism using Multiply and Lambda
def attention_layer(inputs):
    score = Dense(1, activation='tanh')(inputs)  # Compute score for each timestep
    attention_weights = Lambda(lambda x: tf.nn.softmax(x, axis=1))(score)  # Softmax normalization
    attention_output = Multiply()([inputs, attention_weights])  # Apply weights
    return Lambda(lambda x: tf.reduce_sum(x, axis=1))(attention_output)  # Summing weighted outputs

