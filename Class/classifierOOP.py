# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from huggingface_hub import notebook_login, push_to_hub_keras, from_pretrained_keras

transformers.logging.set_verbosity_error()

class DataLoader:
    def __init__(self, topics_path='data/topics.csv', opinions_path='data/opinions.csv'):
        self.df_topics = pd.read_csv(topics_path)
        self.df_opinions = pd.read_csv(opinions_path)
        self.max_length = 256
        self.batch_size = 32
        self.epochs = 2
        self.labels_similarity = ['Effective', 'Adequate', 'Ineffective']
        self.labels_type = ['Claim', 'Evidence', 'Counterclaim', 'Rebuttal']
        self.df_opinions["label_effectiveness"] = self.df_opinions["effectiveness"].apply(
            lambda x: 0 if x == "Effective" else 1 if x == "Adequate" else 2
        )
        self.df_opinions["label_type"] = self.df_opinions["type"].apply(
            lambda x: 0 if x == "Claim" else 1 if x == "Evidence" else 2 if x == 'Counterclaim' else 3
        )
        self.df = pd.merge(self.df_topics[['topic_id', 'text']], self.df_opinions[['topic_id', 'text', 'label_type', 'label_effectiveness']], on='topic_id')
        self.df.rename({'text_x': 'topic', 'text_y': 'opinions'}, axis='columns', inplace=True)

    def tokenize_data(self):
        len_topic_token = self.df['topic'].apply(lambda x: len(x.split()))
        len_opinions_token = self.df['opinions'].apply(lambda x: len(x.split()))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(len_topic_token, bins=20, color='C0', edgecolor='C0')
        axes[0].set_title(f"Topic Token Length (max:{max(len_topic_token)})")
        axes[0].set_xlabel("Length")
        axes[0].set_ylabel("Count")

        axes[1].hist(len_opinions_token, bins=20, color='C0', edgecolor='C0')
        axes[1].set_title(f"Opinions Token Length (max:{max(len_opinions_token)})")
        axes[1].set_xlabel("Length")
        plt.tight_layout()
        plt.show()

        self.df['token_count'] = self.df['opinions'].apply(lambda x: len(x.split()))
        self.df = self.df[self.df['token_count'] <= 256]
        self.df = self.df.drop(columns=['token_count'])
        self.df.reset_index(drop=True)

        df_type = self.df[['topic', 'opinions', 'label_type']]
        y_type = tf.keras.utils.to_categorical(df_type.label_type, num_classes=4)

        df_effectiveness = self.df[['topic', 'opinions', 'label_effectiveness']]
        y_effectiveness = tf.keras.utils.to_categorical(df_effectiveness.label_effectiveness, num_classes=3)

        return df_type, y_type, df_effectiveness, y_effectiveness


class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    def __init__(
            self,
            sentence_pairs,
            labels,
            batch_size=32,
            shuffle=True,
            include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            truncation=True,
            return_tensors="tf",
        )

        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


class BertModelTrainer:
    def __init__(self, len_class, X, y):
        self.len_class = len_class
        self.X = X
        self.y = y
        self.max_length = 256
        self.batch_size = 32
        self.epochs = 2
        self.labels_similarity = ['Effective', 'Adequate', 'Ineffective']
        self.labels_type = ['Claim', 'Evidence', 'Counterclaim', 'Rebuttal']

    def set_model(self):
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            input_ids = tf.keras.layers.Input(
                shape=(self.max_length,), dtype=tf.int32, name="input_ids"
            )
            attention_masks = tf.keras.layers.Input(
                shape=(self.max_length,), dtype=tf.int32, name="attention_masks"
            )
            token_type_ids = tf.keras.layers.Input(
                shape=(self.max_length,), dtype=tf.int32, name="token_type_ids"
            )
            bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
            bert_model.trainable = False

            bert_output = bert_model.bert(
                input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
            )
            sequence_output = bert_output.last_hidden_state
            pooled_output = bert_output.pooler_output
            bi_lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True)
            )(sequence_output)
            avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
            max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
            concat = tf.keras.layers.concatenate([avg_pool, max_pool])
            dropout = tf.keras.layers.Dropout(0.3)(concat)
            output = tf.keras.layers.Dense(self.len_class, activation="softmax")(dropout)
            model = tf.keras.models.Model(
                inputs=[input_ids, attention_masks, token_type_ids], outputs=output
            )

           

class Main:
    def __init__(self):
        self.data_loader = DataLoader()
        self.df_type, self.y_type, self.df_effectiveness, self.y_effectiveness = self.data_loader.tokenize_data()

    def run(self):
        self.train_and_evaluate_effectiveness_model()
        self.train_and_evaluate_type_model()

    def train_and_evaluate_effectiveness_model(self):
        trainer_effectiveness = BertModelTrainer(len(self.data_loader.labels_similarity),
                                                 self.df_effectiveness[["topic", "opinions"]].values.astype("str"),
                                                 self.y_effectiveness)
        model_effectiveness = trainer_effectiveness.set_model()

        test_data_effectiveness = BertSemanticDataGenerator(
            self.df_effectiveness[["topic", "opinions"]].values.astype("str"),
            self.y_effectiveness,
            batch_size=self.data_loader.batch_size,
            shuffle=False,
        )
        model_effectiveness.evaluate(test_data_effectiveness, verbose=1)

        # Save or push to Hugging Face Hub
        # push_to_hub_keras(model_effectiveness, "Bert_Based_effectiveness_classifier_V2")

    def train_and_evaluate_type_model(self):
        trainer_type = BertModelTrainer(len(self.data_loader.labels_type),
                                       self.df_type[["topic", "opinions"]].values.astype("str"),
                                       self.y_type)
        model_type = trainer_type.set_model()

        test_data_type = BertSemanticDataGenerator(
            self.df_type[["topic", "opinions"]].values.astype("str"),
            self.y_type,
            batch_size=self.data_loader.batch_size,
            shuffle=False,
        )
        model_type.evaluate(test_data_type, verbose=1)

        # Save or push to Hugging Face Hub
        # push_to_hub_keras(model_type, "Bert_Based_type_classifier_V2")


if __name__ == "__main__":
    main = Main()
    main.run()