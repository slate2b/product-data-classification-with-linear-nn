"""
Preprocessor

This script is designed to be called from a training script.
The prepare_dataset function splits a CSV file into separate train,
validation, and test splits using the Hugging Face Datasets library.

This file can also be imported as a module and contains the following
function:

    * prepare_dataset - Splits a CSV file using the Datasets library

"""

from datasets import load_dataset

#####
# Adjust as desired
#
_train_percent = 0.70


def prepare_dataset(source_fpath):
    """

    :param source_fpath:
    :return:
    """

    original = load_dataset(
        'csv',
        data_files=source_fpath,
        delimiter=',',
        column_names=['text', 'label'],
        skiprows=1
    )

    #####
    # use train_test_split to create train, validation, and test splits
    #

    # shuffle the original dataset
    original = original.shuffle(seed=1)

    # split the shuffled dataset into train/test splits according to global _train_percent
    dataset = original['train'].train_test_split(train_size=_train_percent)

    # create a new temporary dataset to split the test split into 2 splits
    # this will result in a train and test split here, too
    temporary_test_validation = dataset['test'].train_test_split(train_size=0.5)

    # pop the test split from the main dataset since we will be using the splits below
    dataset.pop('test')

    # create a validation split based on the temporary_test_validation 'train' split
    # this will result in a train, test, and validation split (copy of train)
    temporary_test_validation['validation'] = temporary_test_validation['train']

    # pop the 'train' split from the temporary dataset
    temporary_test_validation.pop('train')

    # update the main dataset to include the new test and validation splits
    dataset.update(temporary_test_validation)

    return dataset
