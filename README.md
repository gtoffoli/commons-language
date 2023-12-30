# commons-language - Open Source Text Analysis Tool

## About the project

*commons-language* is under developent as an extension of the project: https://github.com/gtoffoli/commons , i.e. as a Django app.
It is run as a separate server process, providing services to other web application through HTTP API.
Higher level functions are implemented in a further app, [commons-textanalysis](https://github.com/gtoffoli/commons-textanalysis).  

The full documentation of *commons-language* is in progress. Partially obsolete design ideas can be found at https://www.commonspaces.eu/help/evaluation/ .

*commons-language* was initially created as a clone of [NLPBuddy](https://github.com/eellak/nlpbuddy); it shares with *NLPBuddy* the dependency on the [Spacy](https://spacy.io) library.
The GitHub repository of *NLPBuddy* is still there, but it seems that the live demo is off.
Although *commons-language* exposes its original functionality through HTTP API, we have kept almost unchanged, inside it, the interactive demo of NLPBuddy: you can access it at http://nlp.commonspaces.eu/ .
