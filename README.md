# About

We carried out Sematic Similarity, Classification and Summarization operations using LLMs in the project. A Bert-based LLM was used for Semantic Similarity and Classification, and a Bart-based LLM was used for Summarization.

## Folder Design
If you want to make a quick prediction, go to the predicter.ipynb file, where the models installed on the Huggingface hub are downloaded and predictions are made.
Inside the Class folder, there are Python files in Object Oriented Programming format created for both algorithms.

## Model

https://huggingface.co/necover
## Accuracy

- Accuracy of Classification and Similarity -> 0.8464
- Rogue Score of Summarization -> 0.023027


## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
#For Summarization
input = 'Limiting our usage of any type of motorized vehicle, would not only decrease air pollution but it would also help with adults or children that may not exercise enough.'
output = summarization_pipeline(input)[0]['summary_text']
print(f'Input :{input}\nOutput :{output}')
#Example of output: 
#Input :Limiting our usage of any type of motorized vehicle, would not only decrease air pollution but it would also help with adults or children that may not exercise enough.
#Output : would also help with adults or children that may not exercise enough. Limiting our usage of any type of motorized vehicle, would not only decrease air pollution but it would also improve our health. It would be a good idea to limit the amount of air pollution in the United States, and it would help with the health of adults and children.


#For Classification and Similarity
sentence1 = "Car-free cities seem to be more safe."
sentence2 = "make the streets safer "
check(sentence1, sentence2, model_effectiveness, labels_similarity),check(sentence1, sentence2, model_type, labels_type)
#Example of output: (('Adequate', ' 0.82%'), ('Claim', ' 0.99%'))
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
