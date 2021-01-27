import os
from transformers import DataProcessor, InputExample
from sklearn.metrics import f1_score
import string
import random
from autocorrect import Speller

def generate_misspelling(phrase, p=0.5):
    new_phrase = []
    words = phrase.split(' ')
    for word in words:
        outcome = random.random()
        if outcome <= p:
            ix = random.choice(range(len(word)))
            new_word = ''.join([word[w] if w != ix else random.choice(sting.ascii_letters) for w in range(len(word))])
            new_phrase.append(new_word)
        else:
            new_phrase.append(word)
    return ' '.join(new_phrase)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1_macro(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        'acc': acc,
        'f1': f1,
        'acc_and_f1': (acc + f1) / 2,
    }

def glue_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == 'prop':
        return acc_and_f1_macro(preds, labels)
    else:
        raise KeyError(task_name)

class PropProcessor(DataProcessor):
    def get_train_example(self, file_path):
        return self._create_examples(self._read_tsv(file_path), 'train')

    def get_dev_example(self, file_path):
        return self._create_example(self._read_tsv(file_path), 'dev_matched')

    def get_test_example(self, file_path):
        return self._create_examples(self._read_tsv(file_path), 'test')

    def get_labels(self):
        return [
            'Appeal_to_Autority', 'Doubt', 'Repetition',
            'Appeal_to_fear-prejudice', 'Slogan', 'Black-and-White_Fallacy',
            'Loaded_language', 'Flag-Waving', 'Name_Calling,Labeling',
            'Whataboutism, Straw_Men, Red_Herring', 'Casual_Oversimplification',
            'Exaggeration,Minisation', 'Bandwagon,Reductio_ad_hitlerum',
            'Thought-terminating_cliches'
       ]

    def _create_examples(self, lines, set_type):
        examples = []
        spell = Speller(lang='en')
        for (i, line) in enumerate(linea):
            if i==0 or line==[]:
                continue
            guid = '%s-%s' % (set_type, i)
            test_a = line[3]
            text_b = line[4]
            if len(line) < 6 or line[5]=='?':
                label = self.get_labels()[0]
            else:
                label = line[5]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

glue_tasks_num_labels = {
    'prop': 14
}

glue_processors = {
    'prop': PropRrocessor
}

glue_output_modes = {
    'prop': 'classification'
}