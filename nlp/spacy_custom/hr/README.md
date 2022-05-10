# Prototyping the spaCy language model for Croatian

The code is intended for documentation of work done, results and problems met.

## Language resources used

Please, get information on authors, publishers and licences at the web addresses provided below.

- Inflexional lexicon hrLex 1.3 in the Slovenian Clarin repository -
https://www.clarin.si/repository/xmlui/handle/11356/1232

- conllu corpora UD_Croatian-SET (train, dev, test) listed in the site of universaldependencies.org - 
https://universaldependencies.org/hr/index.html ,
https://github.com/UniversalDependencies/UD_Croatian-SET

- Training corpus hr500k 1.0, listed in the Slovenian Clarin repository -
https://www.clarin.si/repository/xmlui/handle/11356/1183
https://github.com/reldi-data/hr500k/blob/master/hr500k.conllu
https://vukbatanovic.github.io/publication/jtdh_2018_hr/

- The SETimes.HR+ Croatian dependency treebank, listed in the page of the NLP group at unizg.hr (Zagreb) -
http://nlp.ffzg.hr/resources/corpora/setimes-hr/
https://github.com/ffnlp/sethr
http://www.lrec-conf.org/proceedings/lrec2014/pdf/690_Paper.pdf

## Synthetic report on preliminary work done

I don't know the Croatian language, but I wanted to extend to Croatian some text evaluation functions, concerning text complexity and readability, already being developed for English, Italian, Greek and Spanish on top of spaCy in an OS collaborative learning platform, inside an Erasmus+ project.
Since a spaCy language models for Croatian was missing, I've tried to build it myself.

First, I've generated a few components with spaCy 3.0.5 (now I'm working with last version), starting from the *UD_Croatian-SET*.
I have essentially used the basic configuration produced by the *Quickstart Widget* at https://spacy.io/usage/training, and met no problems.
The initial pipeline components were: ['tok2vec', 'tagger', 'morphologizer', 'parser'].
In this step I didn't need to perform any custom preparation of the source data.

Then, I created a *POS-based lemmatizer*, by exploiting the *Inflexional lexicon hrLex 1.3.*
Unlike another lexicon that was mentioned in spaCy issue #4164 (2019), hrLex 1.3 seems to have a convenient licence: "Creative Commons - Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
I've moved along the lines of the Polish lemmatizer, which I had already imitated for Italian.
Since I don't know Croatian, I worked a little blind, without being able to apply any optimization.
Unfortunately, the lookup tables are quite large, possibly because Croatian is a highly inflectional language: the size of lookups.bin is 39874KB; I guess that a rule-based lemmatizer would be much more efficient. Please note also that the lookup tables could be unduly redundand: I've defined the mapping from *look_up POS* to *lexicon POS* in a questionable way, inspired by my experience with the Italian case.
The pipeline components now were: ['lemmatizer', 'tok2vec', 'tagger', 'morphologizer', 'parser'].

I also managed to create a *NER* model. I created it independently, using data from a different source, The *SETimes.HR+* corpus, which contains only sentences with *named entities* (NE); it reached a high performance (about 0.87). But I wasn't able to integrate it with the other models.
Now, *polm* (Paul O'Leary McCann), from *explosion.ai*, has given me this tip, which I will apply soon:
> *It's fine to train the NER model separately and merge it in to the main pipeline afterwards; we do that for many official models. The way to add a component (with its tok2vec) to another pipeline is called "Sourcing Components" and is covered [here](https://spacy.io/usage/processing-pipelines/#sourced-components) in the docs.*

*(you can skip this paragraph)*
At first, to overcome my inability to integrate components from separately trained pipelines, I looked for a corpus including both UD and NE-related tagging. I found one, *Training corpus hr500k 1.0*. In fact, this seems to be the source from which the *UD_Croatian-SET* was derived. With the conversion function *normalize_split_hr500k()*, defined in the *scripts* module (this folder) I managed to derive training and testing data, to convert them to the *.spacy* format and to train all five models. However, while the performance of the other pipeline components has remained practically unchanged, NER  performance has deteriorated significantly (down to about 0.67).

## Re-starting (almost) from scratch

### Python and spaCy versions
After doing some version upgrades with *pip* and *spacy load*, this is the situation:
``` (commons32) C:\commons32>python -m spacy info
←[1m
============================== Info about spaCy ==============================←[0m
spaCy version    3.3.0
Location         C:\commons32\lib\site-packages\spacy
Platform         Windows-10-10.0.19041-SP0
Python version   3.7.3
Pipelines        de_core_news_sm (3.3.0), el_core_news_sm (3.3.0), en_core_web_md (3.3.0), en_core_web_sm (3.3.0), es_core_news_md (3.3.0), es_core_news_sm (3.3.0), fr_core_news_sm (3.3.0), it_core_news_md (3.3.0), it_core_news_sm (3.3.0), lt_core_news_sm (3.3.0), pl_core_news_sm (3.3.0), pt_core_news_sm (3.3.0)
```
NOTES
- *commons32* is the name of the Python virtual environment (venv) in which I executed spaCy CLI commands and custom Python scripts
- *commons32/nlproot* is the base directory of a Django project which uses spaCy for some higher level functionality and provides me with a confortable work environment in contributing to spaCy
- *nlp* is the Django "app", inside nlproot, which exposes to a learning platform, through HTTP API, functions extending spaCy.

### Developing an independent NER model

#### Preparing and validating the data
```(commons32) C:\_Tecnica\AI\CL\spacy\training\hr>python -m spacy convert setimes.hr.v1.conllx-train.iob .\ --converter ner -n 10 --lang hr
ℹ Auto-detected token-per-line NER format
ℹ Grouping every 10 sentences into a document.
✔ Generated output file (600 documents): setimes.hr.v1.conllx-train.spacy
(commons32) C:\_Tecnica\AI\CL\spacy\training\hr>python -m spacy convert setimes.hr.v1.conllx-test.iob .\ --converter ner -n 10 --lang hr
ℹ Auto-detected token-per-line NER format
ℹ Grouping every 10 sentences into a document.
✔ Generated output file (200 documents): setimes.hr.v1.conllx-test.spacy

(commons32) C:\_Tecnica\AI\CL\spacy\training\hr>python -m spacy debug data config_ner.cfg --paths.train ./setimes.hr.v1.conllx-train.spacy --paths.dev ./setimes.hr.v1.conllx-test.spacy
←[1m
============================ Data file validation ============================←[0m
✔ Pipeline can be initialized with data
✔ Corpus is loadable
←[1m
=============================== Training stats ===============================←[0m
Language: hr
Training pipeline: tok2vec, ner
600 training docs
200 evaluation docs
✔ No overlap between training and evaluation data
⚠ Low number of examples to train a new pipeline (600)
←[1m
============================== Vocab & Vectors ==============================←[0m
ℹ 133657 total word(s) in the data (23454 unique)
ℹ No word vectors present in the package
←[1m
========================== Named Entity Recognition ==========================←[0m
ℹ 3 label(s)
0 missing value(s) (tokens with '-' label)
✔ Good amount of examples for all labels
✔ Examples without occurrences available for all labels
✔ No entities consisting of or starting/ending with whitespace
✔ No entities crossing sentence boundaries
←[1m
================================== Summary ==================================←[0m
✔ 7 checks passed
⚠ 1 warning
```
#### Defining the configuration
With the spaCy Quikstart Widget at https://spacy.io/usage/training, generated *base_config_ner.cfg* with parameters: Croatian, ner, CPU, efficiency).
Then, with CLI command *init fill-config*, generated *config_ner.cfg*.

#### Training the NER
```(commons32) C:\_Tecnica\AI\CL\spacy\training\hr>python -m spacy train config_ner.cfg --output ./output_ner --paths.train ./setimes.hr.v1.conllx-train.spacy --paths.dev ./setimes.hr.v1.conllx-test.spacy
ℹ Saving to output directory: output_ner
ℹ Using CPU
←[1m
=========================== Initializing pipeline ===========================←[0m
[2022-05-09 23:44:50,822] [INFO] Set up nlp object from config
[2022-05-09 23:44:50,837] [INFO] Pipeline: ['tok2vec', 'ner']
[2022-05-09 23:44:50,844] [INFO] Created vocabulary
[2022-05-09 23:44:50,846] [INFO] Finished initializing nlp object
[2022-05-09 23:44:54,717] [INFO] Initialized pipeline components: ['tok2vec', 'ner']
✔ Initialized pipeline
←[1m
============================= Training pipeline =============================←[0m
ℹ Pipeline: ['tok2vec', 'ner']
ℹ Initial learn rate: 0.001
E    #       LOSS TOK2VEC  LOSS NER  ENTS_F  ENTS_P  ENTS_R  SCORE
---  ------  ------------  --------  ------  ------  ------  ------
  0       0          0.00     73.29    0.11    0.14    0.08    0.00
  0     200       1590.62   4957.37   66.56   65.72   67.42    0.67
  0     400       1129.61   2680.60   74.62   74.98   74.26    0.75
  1     600        247.21   1742.35   77.93   76.94   78.95    0.78
  1     800        224.66   1188.98   80.82   80.10   81.56    0.81
  1    1000        206.28   1064.53   82.99   82.46   83.53    0.83
  2    1200        203.73   1044.89   84.81   84.14   85.49    0.85
  2    1400        208.00    664.26   84.57   84.06   85.10    0.85
  2    1600        236.28    672.54   83.87   81.93   85.91    0.84
  3    1800        285.85    711.08   86.35   87.72   85.01    0.86
  3    2000        253.17    478.46   85.56   86.87   84.28    0.86
  3    2200        325.64    495.32   86.81   87.09   86.53    0.87
  4    2400        399.58    486.70   86.57   86.95   86.19    0.87
  4    2600        290.55    336.22   86.65   86.95   86.36    0.87
  4    2800       8720.74    392.96   86.50   85.62   87.40    0.87
  5    3000        418.14    377.78   86.34   86.74   85.94    0.86
  5    3200        333.35    252.06   86.53   87.23   85.85    0.87
  5    3400        365.09    258.93   86.18   85.15   87.23    0.86
  6    3600        457.85    300.48   86.55   86.75   86.36    0.87
  6    3800        400.23    218.56   86.27   85.38   87.17    0.86
✔ Saved pipeline to output directory
output_ner\model-last
```
#### Trying to use a language model including only tokenizer and NER

I've copied the output folder *output_ner/model-best* to the *site-packages* directory of my virtual environment and renamed it to *hr_ner_news_sm*. Then, after loading it with the *nlp* Django app (in this *commons-languages* repository), I've made a very simple test.

```(commons32) C:\commons32\nlproot>python manage.py shell
Warning: model nl_core_news_sm not found.
Warning: model hr_core_news_sm not found.
Python 3.7.3 (v3.7.3:ef4ec6ed12, Mar 25 2019, 22:22:05) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
(InteractiveConsole)
>>> from django.conf import settings
>>> nlp = settings.LANGUAGE_MODELS['hr']
>>> nlp
<spacy.lang.hr.Croatian object at 0x000001C330312C88>
>>> nlp.pipeline
[('tok2vec', <spacy.pipeline.tok2vec.Tok2Vec object at 0x000001C3457D6168>), ('ner', <spacy.pipeline.ner.EntityRecognizer object at 0x000001C32E003278>)]
>>> doc = nlp("Kosovo ozbiljno analizira proces privatizacije u svjetlu učestalih pritužbi . Muhamet Brajshori za Southeast European Times iz Prištine -- 21/03/12")
>>> doc.ents
(Kosovo, Muhamet Brajshori, Southeast European Times, Prištine)
>>> [ent.label_ for ent in doc.ents]
['LOC', 'PERS', 'ORG', 'LOC']
>>>
```
### Developing a CORE language model
Now I intended to develop a core language model for HR, exploiting the already existent NER component.

#### Preparing the data

```(commons32) C:\_Tecnica\AI\CL\spacy\training\hr>python -m spacy convert hr_set-ud-train.conllu ./ -l hr -n 10
ℹ Grouping every 10 sentences into a document.
✔ Generated output file (692 documents): hr_set-ud-train.spacy
(commons32) C:\_Tecnica\AI\CL\spacy\training\hr>python -m spacy convert hr_set-ud-test.conllu ./ -l hr -n 10
ℹ Grouping every 10 sentences into a document.
✔ Generated output file (114 documents): hr_set-ud-test.spacy
```
#### Defining the configuration

With the spaCy Quikstart Widget at https://spacy.io/usage/training, generated base_config_core.cfg with parameters: Croatian, tagger morphologizer parser ner, CPU, efficiency)

```(commons32) C:\_Tecnica\AI\CL\spacy\training\hr>python -m spacy init fill-config base_config_core.cfg config_core.cfg
✔ Auto-filled config with all values
✔ Saved config
config_core.cfg
```

#### Validating the data

In running the *debug data* validation, I don't expect that NE-realted data are found.

``` (commons32) C:\_Tecnica\AI\CL\spacy\training\hr>python -m spacy debug data config_core.cfg --paths.train ./hr_set-ud-train.spacy --paths.dev ./hr_set-ud-dev.spacy
←[1m
============================ Data file validation ============================←[0m
✔ Pipeline can be initialized with data
✔ Corpus is loadable
←[1m
=============================== Training stats ===============================←[0m
Language: hr
Training pipeline: tok2vec, tagger, morphologizer, parser, ner
692 training docs
96 evaluation docs
✔ No overlap between training and evaluation data
⚠ Low number of examples to train a new pipeline (692)
←[1m
============================== Vocab & Vectors ==============================←[0m
ℹ 152857 total word(s) in the data (32465 unique)
⚠ 101 misaligned tokens in the training data
⚠ 14 misaligned tokens in the dev data
ℹ No word vectors present in the package
←[1m
========================== Named Entity Recognition ==========================←[0m
ℹ 0 label(s)
0 missing value(s) (tokens with '-' label)
✔ Good amount of examples for all labels
✔ Examples without occurrences available for all labels
✔ No entities consisting of or starting/ending with whitespace
✔ No entities crossing sentence boundaries
←[1m
=========================== Part-of-speech Tagging ===========================←[0m
ℹ 660 label(s) in train data
←[1m
========================= Morphologizer (POS+Morph) =========================←[0m
ℹ 835 label(s) in train data
←[1m
============================= Dependency Parsing =============================←[0m
ℹ Found 6908 sentence(s) with an average length of 22.1 words.
ℹ Found 373 nonprojective train sentence(s)
ℹ 37 label(s) in train data
ℹ 133 label(s) in projectivized train data
⚠ Low number of examples for label 'vocative' (20)
⚠ Low number of examples for label 'dislocated' (7)
⚠ Low number of examples for label 'dep' (7)
⚠ Low number of examples for label 'compound' (8)
⚠ Low number of examples for 88 label(s) in the projectivized dependency trees
used for training. You may want to projectivize labels such as punct before
training in order to improve parser performance.
←[1m
================================== Summary ==================================←[0m
✔ 7 checks passed
⚠ 11 warnings
```

#### Customizing the core configuration with a reference to the NER model

[...]

#### Training the core models

[...]

#### Generating the lookup tables for the Lemmatizer

[...]

#### Integrating the Lemmatizer in the core language model

[...]

## Some references

- spaCy doc - Training Pipelines & Models -
https://spacy.io/usage/training

- spaCy doc - Sourcing components from existing pipelines -
https://spacy.io/usage/processing-pipelines#sourced-components

- spaCy repo - Adding models for new languages master thread #3056 -
https://github.com/explosion/spaCy/discussions/3056

[...]
