# Hakkında

Projede LLM'ler kullanılarak Sematic Similarity, Classification ve Summarization işlemlerini gerçekleştirdik. Semantic Similarity ve Classification işlemi için Bert tabanlı, Summarization için Bart tabanlı bir LLM kullanıldı.

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
