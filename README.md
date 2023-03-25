# text-summarizer


text 파일을 불러와서 (주로 영어로된)  이것을 요약하는 프로그램. 텍스트 붙여넣기로 텍스트 가져오기 가능 <br/> 
A program to import a text file and summarize it (usually in English). Text can be imported by pasting text  <br/> 



<br/> 

![대표](https://github.com/leeseomin/text-summarizer/blob/main/pic/1.png)



  <br/> <br/><br/> 
  
###  Dependency (Tested on an M1 Mac) : cpu version


``` conda install pytorch torchvision torchaudio -c pytorch ```

```pip install pdfminer.six``` 

```
pip install spacy

python -m spacy download en_core_web_sm
```



fairseq install on mac osx

```
git clone https://github.com/pytorch/fairseq

cd fairseq

CFLAGS="-stdlib=libc++" pip install --editable ./
``` 

If the above installation doesn't work on Mac, try installing from conda base.

fairseq install using conda 

```conda install -c conda-forge fairseq```


fairseq install from source 
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```


 <br/><br/> 
 
 
## Key Features


Import text by importing or pasting and summarize it using BART.


텍스트를 불러오거나 붙여넣기해서 가져와서  BART를 사용해서 요약.
 
 
 
## Run Code 

```python v_01.py``` 

 <br/><br/> 



### Limitation

Large files produce strange results when using the text summarization function.



###  To Do


Web app




## Credit

BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension :https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.md


spaCy : https://github.com/explosion/spaCy
