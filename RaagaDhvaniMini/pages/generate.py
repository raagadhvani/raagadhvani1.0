import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import platform
import time
import pathlib
import os

import os
import base64
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href






import streamlit as st
#title of page
st.title("Raagdhvani 🎵")
                                                                                                                                                   


#hiding menu
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

## menu

option = st.selectbox(
    'Select Raaga',
    ('Shankarabharanam','Bhairavi'))


if option=='Shankarabharanam':

    #user input
    song= st.text_input('Enter starting notes')
    #st.write("You can only use notes: s R G m p D N S")

    #upon clicking button
    if st.button(label="Generate Music",key="Generate Music"):

        dataset_file_path = 'skbpallavi.txt'
        text = open(dataset_file_path, mode='r').read()
        # The unique characters in the file
        vocab = sorted(set(text))
        # Map characters to their indices in vocabulary.
        char2index = {char: index for index, char in enumerate(vocab)}
        # Map character indices to characters from vacabulary.
        index2char = np.array(vocab)
        # Convert chars in text to indices.
        text_as_int = np.array([char2index[char] for char in text])
        # The maximum length sentence we want for a single input in characters.
        sequence_length = 239
        examples_per_epoch = len(text) // (sequence_length + 1)
        # Create training dataset.
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        # Generate batched sequences out of the char_dataset.
        sequences = char_dataset.batch(sequence_length + 1, drop_remainder=True)
        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text
        dataset = sequences.map(split_input_target)
        # Batch size.
        BATCH_SIZE = 5

        # Buffer size to shuffle the dataset (TF data is designed to work
        # with possibly infinite sequences, so it doesn't attempt to shuffle
        # the entire sequence in memory. Instead, it maintains a buffer in
        # which it shuffles elements).
        BUFFER_SIZE = 5

        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        # Let's do a quick detour and see how Embeding layer works.
        # It takes several char indices sequences (batch) as an input.
        # It encodes every character of every sequence to a vector of tmp_embeding_size length.
        tmp_vocab_size = 10
        tmp_embeding_size = 5
        tmp_input_length = 8
        tmp_batch_size = 2

        tmp_model = tf.keras.models.Sequential()
        tmp_model.add(tf.keras.layers.Embedding(
        input_dim=tmp_vocab_size,
        output_dim=tmp_embeding_size,
        input_length=tmp_input_length
        ))
        # The model will take as input an integer matrix of size (batch, input_length).
        # The largest integer (i.e. word index) in the input should be no larger than 9 (tmp_vocab_size).
        # Now model.output_shape == (None, 10, 64), where None is the batch dimension.
        tmp_input_array = np.random.randint(
        low=0,
        high=tmp_vocab_size,
        size=(tmp_batch_size, tmp_input_length)
        )
        tmp_model.compile('rmsprop', 'mse')
        tmp_output_array = tmp_model.predict(tmp_input_array)
        # Length of the vocabulary in chars.
        vocab_size = len(vocab)

        # The embedding dimension.
        embedding_dim = 256

        # Number of RNN units.
        rnn_units = 1024
        def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
            model = tf.keras.models.Sequential()

            model.add(tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            batch_input_shape=[batch_size, None]
            ))

            model.add(tf.keras.layers.LSTM(
            units=rnn_units,
            return_sequences=True,
            stateful=True,
            recurrent_initializer=tf.keras.initializers.GlorotNormal()
            ))

            model.add(tf.keras.layers.Dense(vocab_size))
        
            return model
        
        model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)
        for input_example_batch, target_example_batch in dataset.take(1):
            example_batch_predictions = model(input_example_batch)
        # Quick overview of how tf.random.categorical() works.

        # logits is 2-D Tensor with shape [batch_size, num_classes].
        # Each slice [i, :] represents the unnormalized log-probabilities for all classes.
        # In the example below we say that the probability for class "0" is low but the
        # probability for class "2" is much higher.
        tmp_logits = [
        [-0.95, 0, 0.95],
        ];

        # Let's generate 5 samples. Each sample is a class index. Class probabilities 
        # are being taken into account (we expect to see more samples of class "2").
        tmp_samples = tf.random.categorical(
            logits=tmp_logits,
            num_samples=5
        )
        sampled_indices = tf.random.categorical(
        logits=example_batch_predictions[0],
        num_samples=1
        )
        sampled_indices = tf.squeeze(
        input=sampled_indices,
        axis=-1).numpy()
        # An objective function.
        # The function is any callable with the signature scalar_loss = fn(y_true, y_pred).
        def loss(labels, logits):
            return tf.keras.losses.sparse_categorical_crossentropy(
            y_true=labels,
            y_pred=logits,
            from_logits=True
            )
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
        optimizer=adam_optimizer,
        loss=loss
        )
        # Directory where the checkpoints will be saved.
        checkpoint_dir = 'tmp/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')

        checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True
        )

        ### 40 epochs
        EPOCHS=5
        history = model.fit(
    x=dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint_callback
    ]
    )
        tf.train.latest_checkpoint(checkpoint_dir)
        simplified_batch_size = 1

        model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

        model.build(tf.TensorShape([simplified_batch_size, None]))

        # num_generate
        # - number of characters to generate.
        #
        # temperature
        # - Low temperatures results in more predictable text.
        # - Higher temperatures results in more surprising text.
        # - Experiment to find the best setting.
        # changed num_generate from 239 to 20 and temp from 1.0 to 0.7
        def generate_text(model, start_string, num_generate = 20, temperature=0.7):
            # Evaluation step (generating text using the learned model)

            # Converting our start string to numbers (vectorizing).
            input_indices = [char2index[s] for s in start_string]
            input_indices = tf.expand_dims(input_indices, 0)

            # Empty string to store our results.
            text_generated = []

            # Here batch size == 1.
            model.reset_states()
            for char_index in range(num_generate):
                predictions = model(input_indices)
                # remove the batch dimension
                predictions = tf.squeeze(predictions, 0)

                # Using a categorical distribution to predict the character returned by the model.
                predictions = predictions / temperature
                predicted_id = tf.random.categorical(
                predictions,
                num_samples=1
                )[-1,0].numpy()

                # We pass the predicted character as the next input to the model
                # along with the previous hidden state.
                input_indices = tf.expand_dims([predicted_id], 0)

                text_generated.append(index2char[predicted_id])

            return (start_string + ''.join(text_generated))



        

        # Generate the text with default temperature (1.0).
        resultstring=generate_text(model, start_string=song, temperature=0.7)
        rs=str(resultstring)


        ## printing output
        st.write("Predicted result is: ")
        st.write(rs)
        rs=rs.replace(' ','').replace('\n','')


    
        import numpy as np
        from pydub import AudioSegment
        from datetime import datetime
        import streamlit as st

        def generate_wave(note, duration, amplitude=0.5, sample_rate=44100):
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            wave = amplitude * np.sin(2 * np.pi * note * t)
            # Normalize the wave to fit within the range [-1, 1]
            wave = np.int16(wave * 32767)
            return wave

        def create_wav_file(notes, durations, output_file, amplitude=0.8, sample_rate=44100, sample_width=2):
            combined = AudioSegment.silent(duration=0)
            
            for note, duration in zip(notes, durations):
                wave = generate_wave(note, duration, amplitude, sample_rate)
                audio_segment = AudioSegment(
                    wave.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=sample_width,
                    channels=1
                )
                combined += audio_segment

            combined.export(output_file, format="wav")

        def get_frequency(note):
            note_mapping = {
                's': 261.63,
                'r': 293.66,
                'g': 329.63,
                'm': 349.23,
                'p': 392.00,
                'd': 440.00,
                'n': 493.38
            }
            return note_mapping[note.lower()] * (2 if note.isupper() else 1)

        def generate_notes_array(result_string):
            notes = []
            durations = []
            for note in result_string:
                if note.isalpha():
                    frequency = get_frequency(note)
                    notes.append(frequency)
                    durations.append(0.5)  # Assuming a default duration of 0.5 seconds for each note

            return notes, durations

        
        # Example usage:
        result_string = rs
        notes, durations = generate_notes_array(result_string)

        # Generate timestamp for unique file name
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d-%H%M%S")
        output_file = "genskb"+dt_string+".wav"
        create_wav_file(notes, durations, output_file, amplitude=0.8)
        # Get the current date and time
        

        

        st.audio(output_file, format="audio/wav")
        from datetime import datetime

        
        
        st.markdown(get_binary_file_downloader_html("genskb"+dt_string+".wav", 'WAV'), unsafe_allow_html = True)

elif option=='Bhairavi':

    #user input
    song= st.text_input('Enter starting notes')
    #st.write("You can only use notes: s R G m p D N S")

    #upon clicking button
    if st.button(label="Generate Music",key="Generate Music"):

        dataset_file_path = 'bhpallavi.txt'
        text = open(dataset_file_path, mode='r').read()
        # The unique characters in the file
        vocab = sorted(set(text))
        # Map characters to their indices in vocabulary.
        char2index = {char: index for index, char in enumerate(vocab)}
        # Map character indices to characters from vacabulary.
        index2char = np.array(vocab)
        # Convert chars in text to indices.
        text_as_int = np.array([char2index[char] for char in text])
        # The maximum length sentence we want for a single input in characters.
        sequence_length = 239
        examples_per_epoch = len(text) // (sequence_length + 1)
        # Create training dataset.
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        # Generate batched sequences out of the char_dataset.
        sequences = char_dataset.batch(sequence_length + 1, drop_remainder=True)
        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text
        dataset = sequences.map(split_input_target)
        # Batch size.
        BATCH_SIZE = 5

        # Buffer size to shuffle the dataset (TF data is designed to work
        # with possibly infinite sequences, so it doesn't attempt to shuffle
        # the entire sequence in memory. Instead, it maintains a buffer in
        # which it shuffles elements).
        BUFFER_SIZE = 5

        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        # Let's do a quick detour and see how Embeding layer works.
        # It takes several char indices sequences (batch) as an input.
        # It encodes every character of every sequence to a vector of tmp_embeding_size length.
        tmp_vocab_size = 10
        tmp_embeding_size = 5
        tmp_input_length = 8
        tmp_batch_size = 2

        tmp_model = tf.keras.models.Sequential()
        tmp_model.add(tf.keras.layers.Embedding(
        input_dim=tmp_vocab_size,
        output_dim=tmp_embeding_size,
        input_length=tmp_input_length
        ))
        # The model will take as input an integer matrix of size (batch, input_length).
        # The largest integer (i.e. word index) in the input should be no larger than 9 (tmp_vocab_size).
        # Now model.output_shape == (None, 10, 64), where None is the batch dimension.
        tmp_input_array = np.random.randint(
        low=0,
        high=tmp_vocab_size,
        size=(tmp_batch_size, tmp_input_length)
        )
        tmp_model.compile('rmsprop', 'mse')
        tmp_output_array = tmp_model.predict(tmp_input_array)
        # Length of the vocabulary in chars.
        vocab_size = len(vocab)

        # The embedding dimension.
        embedding_dim = 256

        # Number of RNN units.
        rnn_units = 1024
        def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
            model = tf.keras.models.Sequential()

            model.add(tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            batch_input_shape=[batch_size, None]
            ))

            model.add(tf.keras.layers.LSTM(
            units=rnn_units,
            return_sequences=True,
            stateful=True,
            recurrent_initializer=tf.keras.initializers.GlorotNormal()
            ))

            model.add(tf.keras.layers.Dense(vocab_size))
        
            return model
        
        model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)
        for input_example_batch, target_example_batch in dataset.take(1):
            example_batch_predictions = model(input_example_batch)
        # Quick overview of how tf.random.categorical() works.

        # logits is 2-D Tensor with shape [batch_size, num_classes].
        # Each slice [i, :] represents the unnormalized log-probabilities for all classes.
        # In the example below we say that the probability for class "0" is low but the
        # probability for class "2" is much higher.
        tmp_logits = [
        [-0.95, 0, 0.95],
        ];

        # Let's generate 5 samples. Each sample is a class index. Class probabilities 
        # are being taken into account (we expect to see more samples of class "2").
        tmp_samples = tf.random.categorical(
            logits=tmp_logits,
            num_samples=5
        )
        sampled_indices = tf.random.categorical(
        logits=example_batch_predictions[0],
        num_samples=1
        )
        sampled_indices = tf.squeeze(
        input=sampled_indices,
        axis=-1).numpy()
        # An objective function.
        # The function is any callable with the signature scalar_loss = fn(y_true, y_pred).
        def loss(labels, logits):
            return tf.keras.losses.sparse_categorical_crossentropy(
            y_true=labels,
            y_pred=logits,
            from_logits=True
            )
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
        optimizer=adam_optimizer,
        loss=loss
        )
        # Directory where the checkpoints will be saved.
        checkpoint_dir = 'tmp/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')

        checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True
        )

        ### 40 epochs
        EPOCHS=5
        history = model.fit(
    x=dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint_callback
    ]
    )
        tf.train.latest_checkpoint(checkpoint_dir)
        simplified_batch_size = 1

        model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

        model.build(tf.TensorShape([simplified_batch_size, None]))

        # num_generate
        # - number of characters to generate.
        #
        # temperature
        # - Low temperatures results in more predictable text.
        # - Higher temperatures results in more surprising text.
        # - Experiment to find the best setting.
        # changed num_generate from 239 to 20 and temp from 1.0 to 0.7
        def generate_text(model, start_string, num_generate = 20, temperature=0.7):
            # Evaluation step (generating text using the learned model)

            # Converting our start string to numbers (vectorizing).
            input_indices = [char2index[s] for s in start_string]
            input_indices = tf.expand_dims(input_indices, 0)

            # Empty string to store our results.
            text_generated = []

            # Here batch size == 1.
            model.reset_states()
            for char_index in range(num_generate):
                predictions = model(input_indices)
                # remove the batch dimension
                predictions = tf.squeeze(predictions, 0)

                # Using a categorical distribution to predict the character returned by the model.
                predictions = predictions / temperature
                predicted_id = tf.random.categorical(
                predictions,
                num_samples=1
                )[-1,0].numpy()

                # We pass the predicted character as the next input to the model
                # along with the previous hidden state.
                input_indices = tf.expand_dims([predicted_id], 0)

                text_generated.append(index2char[predicted_id])

            return (start_string + ''.join(text_generated))



        

        # Generate the text with default temperature (1.0).
        resultstring=generate_text(model, start_string=song, temperature=0.7)
        rs=str(resultstring)


        ## printing output
        st.write("Predicted result is: ")
        st.write(rs)
        rs=rs.replace(' ','').replace('\n','')


    
        import numpy as np
        from pydub import AudioSegment
        from datetime import datetime
        import streamlit as st

        def generate_wave(note, duration, amplitude=0.5, sample_rate=44100):
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            wave = amplitude * np.sin(2 * np.pi * note * t)
            # Normalize the wave to fit within the range [-1, 1]
            wave = np.int16(wave * 32767)
            return wave

        def create_wav_file(notes, durations, output_file, amplitude=0.8, sample_rate=44100, sample_width=2):
            combined = AudioSegment.silent(duration=0)
            
            for note, duration in zip(notes, durations):
                wave = generate_wave(note, duration, amplitude, sample_rate)
                audio_segment = AudioSegment(
                    wave.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=sample_width,
                    channels=1
                )
                combined += audio_segment

            combined.export(output_file, format="wav")

        def get_frequency(note):
            note_mapping = {
                's': 261.63,
                'r': 277.18,
                'g': 311.13,
                'm': 369.99,
                'p': 392.00,
                'd': 415.40,
                'n': 446.16
            }
            return note_mapping[note.lower()] * (2 if note.isupper() else 1)

        def generate_notes_array(result_string):
            notes = []
            durations = []
            for note in result_string:
                if note.isalpha():
                    frequency = get_frequency(note)
                    notes.append(frequency)
                    durations.append(0.5)  # Assuming a default duration of 0.5 seconds for each note

            return notes, durations

        
        # Example usage:
        result_string = rs
        notes, durations = generate_notes_array(result_string)

        # Generate timestamp for unique file name
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d-%H%M%S")
        output_file = "genbh"+dt_string+".wav"
        create_wav_file(notes, durations, output_file, amplitude=0.8)
        # Get the current date and time
        

        

        st.audio(output_file, format="audio/wav")
        from datetime import datetime

        
        
        st.markdown(get_binary_file_downloader_html("genbh"+dt_string+".wav", 'WAV'), unsafe_allow_html = True)