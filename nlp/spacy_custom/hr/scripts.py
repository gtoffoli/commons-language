from random import randint

def settimes_to_iob(in_path):
    """ From the conllx (extended conllu) format of the SETimes.HR treebank,
        extract only the form and iob fields to be used as NER training data
        and split them between train and eval files in the 3:1 ratio.
        see: https://universaldependencies.org/docs/format.html 
             http://nlp.ffzg.hr/resources/corpora/setimes-hr/
             https://www.clarin.si/repository/xmlui/handle/11356/1183 """
    train_path = in_path+'-train.iob'
    eval_path = in_path+'-eval.iob'
    choices = []
    empty = True
    with open(in_path, encoding='utf8') as infile:
        with open(train_path, 'w', encoding='utf8') as trainfile:
            with open(eval_path, 'w', encoding='utf8') as evalfile:
                if not choices:
                    choices = [trainfile, trainfile, trainfile, evalfile]
                line = infile.readline()
                while(line):
                    if empty:
                        outfile = choices[randint(0, 3)]
                    fields = line.replace('\n','').split('\t')
                    if len(fields) > 1:
                        id, form, lemma, upos, xpos, iob, head, deprel, deps, misc = fields 
                        line = '\t'.join([form, iob])+'\n'
                        empty = False
                    else:
                        empty = True
                    outfile.write(line)
                    line = infile.readline()
        trainfile.close()
        evalfile.close()
    infile.close()

CONLLU_DICT = {
    'ID': 0, # Word index
    'FORM': 1, # Word form or punctuation symbol.
    'LEMMA': 2, # Lemma or stem of word form.
    'UPOS': 3, # Universal part-of-speech tag.
    'XPOS': 4, # Language-specific part-of-speech tag; underscore if not available.
    'FEATS': 5, # List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
    'HEAD': 6, # Head of the current word, which is either a value of ID or zero (0).
    'DEPREL': 7, # Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
    'DEPS': 8, # Enhanced dependency graph in the form of a list of head-deprel pairs.
    'MISC': 9, # Any other annotation.
    'IOB': 9, # annotation for NER.
}

def normalize_split_hr500k(in_path):
    """ From the conll format of the hr500k.conllu treebank,
        remove the comment lines and keep only sentences with UD annotation, 
        split them between train and test files in the 3:1 ratio,
        extract from them the first 10 columns
        see: https://github.com/reldi-data/hr500k/blob/master/hr500k.conllu """
    file_ext = '.'+in_path.split('.')[-1]
    file_name = in_path.replace(file_ext, '')
    train_path = file_name+'-train'+file_ext
    test_path = file_name+'-test'+file_ext
    choices = []
    empty = True
    skip = False
    with open(in_path, encoding='utf8') as infile:
        with open(train_path, 'w', encoding='utf8') as trainfile:
            outfile = trainfile
            with open(test_path, 'w', encoding='utf8') as testfile:
                if not choices:
                    choices = [trainfile, trainfile, trainfile, testfile]
                line = infile.readline()
                while(line):
                    if line.startswith('#'):
                        if line.startswith('# sent_id'):
                            outfile = choices[randint(0, 3)]
                        elif not line.startswith('# text'):
                            pass
                        line = infile.readline()
                        continue
                    else:
                        fields = line.replace('\r','').replace('\n','').split('\t')
                        if len(fields) >= 10:
                            if fields[CONLLU_DICT['HEAD']]=='_' or fields[CONLLU_DICT['DEPREL']]=='_':
                                skip = True
                            else:
                                misc_list = fields[CONLLU_DICT['MISC']].split('|')
                                iob = 'O'
                                for item in misc_list:
                                    if item.startswith('NER='):
                                        iob = item.replace('NER=', '').upper()
                                        break
                                fields = fields[:CONLLU_DICT['MISC']]+[iob]+fields[CONLLU_DICT['MISC']+1:]
                                line = '\t'.join(fields)+'\n'
                            empty = False
                        else:
                            empty = True
                    if empty or not skip:
                        outfile.write(line)
                    if empty:
                        skip = False
                    line = infile.readline()
        trainfile.close()
        testfile.close()
    infile.close()
