
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from datasets import load_metric
from tqdm import tqdm
import torch

class SummarizationModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.rouge_metric = load_metric('rouge')

    def load_data(self, topics_path, opinions_path, conclusions_path):
        topics = pd.read_csv(topics_path)
        opinions = pd.read_csv(opinions_path, error_bad_lines=False)
        conclusions = pd.read_csv(conclusions_path)
        return topics, opinions, conclusions

    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(example_batch['topic'] , max_length = 100, truncation = True )
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], max_length = 260, truncation = True )

        return {
            'input_ids' : input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def preprocess_data(self, topics, opinions, conclusions):
        train_concluions = pd.merge(topics, conclusions[['id','topic_id','text']], on='topic_id')
        train_concluions.rename({'text_x':'topic','text_y':'summary'}, axis='columns', inplace=True)
        return train_concluions

    def train_model(self, train_concluions_pt):
        seq2seq_data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        trainer_args = TrainingArguments(
            output_dir='conclusions', num_train_epochs=1, warmup_steps=500,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            weight_decay=0.01, logging_steps=10,
            evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
            gradient_accumulation_steps=16
        )
        trainer = Trainer(model=self.model, args=trainer_args,
                          tokenizer=self.tokenizer, data_collator=seq2seq_data_collator,
                          train_dataset=train_concluions_pt)
        trainer.train()

    def evaluate_model(self, test):
        score = self.calculate_metric_on_test_ds(test)
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
        print(pd.DataFrame(rouge_dict, index=['test']))

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i: i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, batch_size=8):
        article_batches = list(self.generate_batch_sized_chunks(dataset['topic'].tolist(), batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset['summary'].tolist(), batch_size))

        for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
            inputs = self.tokenizer(article_batch, max_length=512, truncation=True,
                                    padding="max_length", return_tensors="pt")
            summaries = self.model.generate(input_ids=inputs["input_ids"].to(self.device),
                                           attention_mask=inputs["attention_mask"].to(self.device),
                                           length_penalty=0.8, num_beams=8, max_length=128)

            decoded_summaries = [self.tokenizer.decode(s, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=True) for s in summaries]
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]

            self.rouge_metric.add_batch(predictions=decoded_summaries, references=target_batch)
        return self.rouge_metric.compute()



    def save_model(self, output_dir):
        # Modeli kaydetme işlemini buraya ekleyebilirsiniz.
        #notebook_login()
        #self.model.model.push_to_hub("FTBartBasedModel4concluions")
        #self.model.tokenizer.push_to_hub("FTBartBasedModelTokenizer4concluions")
        pass

if __name__ == "__main__":
    model_name = "facebook/bart-large-cnn"
    summarization_model = SummarizationModel(model_name)
    topics, opinions, conclusions = summarization_model.load_data('data/topics.csv', 'data/opinions.csv', 'data/conclusions.csv')
    train_concluions = summarization_model.preprocess_data(topics, opinions, conclusions)
    train_concluions_pt = train_concluions.apply(summarization_model.convert_examples_to_features, axis=1)
    summarization_model.train_model(train_concluions_pt)

    train, test = train_test_split(train_concluions, test_size=0.2, random_state=42)
    summarization_model.evaluate_model(test)
    summarization_model.save_model("output_directory")
