**Code written in prototyping the spaCy language model for Croatian **

The code is intended for documentation of work done, results and problems met.

** Language resources used **

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

** Synthetic report on work done **

I don't know the Croatian language, but I wanted to extend to Croatian some text evaluation functions, concerning text complexity and readability, already being developed for English, Italian, Greek and Spanish on top of spaCy in an OS collaborative learning platform, inside an Erasmus+ project.
Since a spaCy language models for Croatian was missing, I've tried to build it myself.

First, I've generated a few components with spaCy 3.0.5 (now I'm working with last version), starting from the UD_Croatian-SET.
I have essentially used the basic configuration produced by the "quickstart widget" at https://spacy.io/usage/training, and met no problems.
The initial pipeline components were: ['tok2vec', 'tagger', 'morphologizer', 'parser'].
In this step I didn't need to perform any custom preparation of the source data.

Then, I created a POS-based lemmatizer, by exploiting the Inflexional lexicon hrLex 1.3.
Unlike another lexicon that was mentioned in spaCy issue #4164 (2019), hrLex 1.3 seems to have a convenient licence: "Creative Commons - Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
I've moved along the lines of the Polish lemmatizer, which I had already imitated for Italian.
Since I don't know Croatian, I worked a little blind, without being able to apply any optimization.
Unfortunately, the lookup tables are quite large, possibly because Croatian is a highly inflectional language: the size of lookups.bin is 39874KB; I guess that a rule-based lemmatizer would be much more efficient.
The pipeline components now were: ['lemmatizer', 'tok2vec', 'tagger', 'morphologizer', 'parser'].

[...]
