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

def normalize_split_hr500k(in_path):
    """ From the conll format of the hr500k treebank,
        remove the comment lines,
        extract only the first 11 columns (10 + iob) to be used as training data
        and keep only sentences just after comment # sent_id = train.. or # sent_id = test.. 
        # and split them between train and eval files in the 3:1 ratio.
        see: https://www.clarin.si/repository/xmlui/handle/11356/1183 """
    file_ext = '.'+in_path.split('.')[-1]
    file_name = in_path.replace(file_ext, '')
    train_path = file_name+'-train'+file_ext
    test_path = file_name+'-test'+file_ext
    # choices = []
    outfile = None
    empty = True
    with open(in_path, encoding='utf8') as infile:
        with open(train_path, 'w', encoding='utf8') as trainfile:
            with open(test_path, 'w', encoding='utf8') as testfile:
                """
                if not choices:
                    choices = [trainfile, trainfile, trainfile, evalfile]
                """
                line = infile.readline()
                while(line):
                    if empty:
                        if line.startswith('#'):
                            if line.startswith('# sent_id = train'):
                                outfile = trainfile
                            elif line.startswith('# sent_id = test'):
                                outfile = testfile
                            line = infile.readline()
                            continue
                    fields = line.replace('\n','').split('\t')
                    if len(fields) > 1:
                        line = '\t'.join(fields[:11])+'\n'
                        empty = False
                    else:
                        empty = True
                    if outfile:
                        outfile.write(line)
                    if empty:
                        outfile = None
                    line = infile.readline()
        trainfile.close()
        testfile.close()
    infile.close()
