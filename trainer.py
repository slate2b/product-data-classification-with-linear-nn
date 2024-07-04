"""
Trainer

This is the main script in a project designed to test the effectiveness
of a simple sequential model for the use-case of classifying a product
record within a product taxonomy.

Script begins by asking user whether they want to perform a test run
using a tiny version of the dataset.  If user declines, then script
will perform fine-tuning operations using the full dataset.

Be sure to install the following libraries prior to use:

    datasets
    torch
    torchtext

This file can also be imported as a module and contains the following
class:

    * TextClassificationModel - Custom model class which inherits from torch.nn.Module

And the following functions:

    * yield_tokens - Tokenizes text using the given tokenizer
    * build_iter - Builds a list of tuples containing the labels and text from a data list
    * init_iterators - Creates iterable lists for the dataset splits
    * build_vocab - Builds a custom vocabulary based on the given tokenizer and text data
    * init_pipelines - Creates text and label pipelines to be used with the DataLoaders
    * collate_batch - Collate function to be used with Torch DataLoaders
    * init_dataloaders - Creates Torch DataLoaders for each dataset split (train, valid, and test)
    * train - Perform a single epoch of training
    * evaluate - Uses the given dataset split to evaluate the trained model
    * train_model - Manages training, evaluating, and saving the model
    * use_tiny_dataset - executes training operations using a tiny version of the dataset
    * use_full_dataset - executes training operations using the full dataset

    Original dataset taken from https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset/data

"""

import torch
from datasets import DatasetDict
from preprocessor import prepare_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch import nn
import time
from torchtext.data.functional import to_map_style_dataset
import torch.nn.functional as F

source_filepath = './source_data/amazon-products-text-and-label_ids.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global Variables
_epochs = 10
_learning_rate = .0001
_batch_size = 128
_num_batches = 0
_num_labels = 20
_embed_size = 1024
_criterion = torch.nn.CrossEntropyLoss()
train_iter = []
validation_iter = []
test_iter = []
text_pipeline = "initial_placeholder_value"
label_pipeline = "initial_placeholder_value"
train_dataloader = "initial_placeholder_value"
valid_dataloader = "initial_placeholder_value"
test_dataloader = "initial_placeholder_value"


def yield_tokens(tknzr, data_iter):
    """
    Tokenizes text using the given tokenizer.

    yield: The tokenized text

    :param tknzr: (TorchText Tokenizer): The tokenizer to be used
    :param data_iter: (list): An iterable list containing the data to be tokenized
    """
    for _, text in data_iter:
        yield tknzr(text)


def build_iter(data_list):
    """
    Builds a list of tuples containing the labels and text from a data list.

    :param data_list: (list): A list version of the dataset split
    :return iterable_list (list): list of tuples containing the labels and text from the given df
    """

    iterable_list = []

    for i in range(len(data_list.index)):
        label = data_list['label'].iloc[i]
        text = data_list['text'].iloc[i]
        iter_tuple = (label, text)
        iterable_list.append(iter_tuple)

    return iterable_list


def init_iterators(trn_list, val_list, tst_list):
    """
    Creates iterable lists (lists of tuples) for the train,
    validation, and test splits.

    :param trn_list: (list): Data list for the training split
    :param val_list: (list): Data list for the validation split
    :param tst_list: (list): Data list for the test split
    :return None
    """

    global train_iter
    global validation_iter
    global test_iter

    # Create iterable lists for each dataset split
    train_iter = build_iter(trn_list)
    validation_iter = build_iter(val_list)
    test_iter = build_iter(tst_list)

    return


def build_vocab(tknizer, data_iter):
    """
    Builds a custom vocabulary based on the given tokenizer and text data.

    :param tknizer: (TorchText Tokenizer): The tokenizer to be used
    :param data_iter: (list of tuples): An iterable list containing the text data
    :return vocab (TorchText Vocab)
    """

    # Build the vocabulary using the tokenizer defined in global variables
    vocab = build_vocab_from_iterator(yield_tokens(tknizer, data_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    return vocab


def init_pipelines(vcb, tknzr):
    """
    Creates text and label pipelines to be used with the DataLoaders.

    :param vcb: (TorchText Vocab): The vocabulary to be used
    :param tknzr: (TorchText Tokenizer): The tokenizer to be used
    :return None
    """
    # Create pipelines for text processing
    global text_pipeline
    global label_pipeline

    text_pipeline = lambda x: vcb(tknzr(x))
    label_pipeline = lambda x: int(x)

    return


def collate_batch(batch):
    """
    Collate function to be used with Torch DataLoaders

    :param batch: The batch to be collated
    :return label_list, text_list, offsets (Torch Tensors): The batched labels, text, and offsets
    """

    global text_pipeline
    global label_pipeline

    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    labels = torch.tensor(label_list, dtype=torch.int64)
    text = torch.cat(text_list)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

    return labels.to(device), text.to(device), offsets.to(device)


def init_dataloaders():
    """
    Creates Torch DataLoaders for each dataset split (train, valid, and test).

    :return None
    """
    global train_dataloader
    global valid_dataloader
    global test_dataloader
    global train_iter
    global validation_iter
    global test_iter

    # Convert the dataset splits to map style datasets
    train_dataset = to_map_style_dataset(train_iter)
    validation_dataset = to_map_style_dataset(validation_iter)
    test_dataset = to_map_style_dataset(test_iter)

    # Create DataLoaders to manage batches to send to the model
    train_dataloader = DataLoader(
        train_dataset, batch_size=_batch_size, shuffle=False, collate_fn=collate_batch
    )
    valid_dataloader = DataLoader(
        validation_dataset, batch_size=_batch_size, shuffle=False, collate_fn=collate_batch
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=_batch_size, shuffle=False, collate_fn=collate_batch
    )

    return


class TextClassificationModel(nn.Module):
    """
    A class for the custom model. This class inherits from torch.nn.Module.

    Attributes:
         embedding (nn.EmbeddingBag): Embedding layer
         fc1 (nn.Linear): Linear layer
         do1 (nn.Dropout): Dropout layer
         fc2 (nn.Linear): Linear layer
         do2 (nn.Dropout): Dropout layer
         output (nn.Linear): Linear layer (Output)
    """
    def __init__(self, vocab_size, embed_dim, num_class):
        """
        Initializes a TextClassificationModel object.

        :param vocab_size: (int): The number of tokens in the vocabulary
        :param embed_dim: (int): The embedding dimensions for the model
        :param num_class: (int): The number of classes (labels)
        """
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.do1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(embed_dim, embed_dim // 2)
        self.do2 = nn.Dropout(0.3)
        self.output = nn.Linear(embed_dim // 2, num_class)
        self.init_weights()

    def init_weights(self):
        """
        Initializes weights for the embedding and linear layers

        :return None
        """
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)
        self.output.bias.data.zero_()

    def forward(self, text, offsets):
        """
        The computation to be performed during each forward pass.

        :param text: The text generated from the DataLoader
        :param offsets: The offsets generated from the DataLoader
        :return output: The predicted labels
        """
        embedded = self.embedding(text, offsets)
        f1 = F.relu(self.fc1(embedded))
        d1 = self.do1(f1)
        f2 = F.relu(self.fc2(d1))
        d2 = self.do2(f2)
        output = self.output(d2)
        return output


def train(model, loader, optimizer, epch):
    """
    Perform a single epoch of training.

    :param model: (nn.Module): The model to be trained
    :param loader: (Torch DataLoader): The DataLoader to generate training batches
    :param optimizer: (Torch Optimizer): The Optimizer to use for training
    :param epch: (int): The current epoch
    :return:
    """

    global _num_batches

    model.train()  # set model to training mode
    total_acc, total_count = 0, 0
    log_interval = _num_batches
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(loader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = _criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| train accuracy {:8.3f}".format(
                    epch, idx, len(loader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(model, loader):
    """
    Uses the given dataset split to evaluate the trained model.

    :param model: (nn.Module): The model to be evaluated
    :param loader: (Torch DataLoader): The DataLoader to generate training batches
    :return avg_acc, loss
    """

    model.eval()  # set model to evaluation mode
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(loader):
            predicted_label = model(text, offsets)
            loss = _criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            avg_acc = total_acc / total_count
    return avg_acc, loss


def train_model(model, scheduler, optimizer):
    """
    Manages training, evaluating, and saving the model.

    :param model: (nn.Module): The model to be trained
    :param scheduler: (Torch StepLR): The learning rate scheduler
    :param optimizer: (Torch Optimizer): The optimizer to be used for training
    :return None
    """

    global train_dataloader
    global valid_dataloader
    global test_dataloader

    # Initialize total accuracy to None
    total_accu = None

    # Train and run a validation eval for each epoch
    for epoch in range(1, _epochs + 1):
        epoch_start_time = time.time()
        train(model=model, loader=train_dataloader, optimizer=optimizer, epch=epoch)
        accu_val, loss_val = evaluate(model, valid_dataloader)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print("-" * 89)
        print(
            "| end of epoch {:3d} | time: {:6.2f}s | "
            "valid accuracy {:8.3f} | valid loss {:14.4f}".format(
                epoch, time.time() - epoch_start_time, accu_val, loss_val
            )
        )
        print("-" * 89)

    # Perform an evaluation based on the test dataset split
    print("Checking the results of test dataset.")
    accu_test, loss_test = evaluate(model, test_dataloader)
    print("test accuracy {:8.3f} | test loss {:8.4f}".format(accu_test, loss_test))

    # Print model's state_dict
    print("\nModel's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print("\nSaving model...\n")

    model_save_path = './product-data-classification-linear-model.pt'
    optimizer_save_path = './product-data-classification-linear-optimizer.pt'
    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)

    return


def use_tiny_dataset(dataset):
    """
    Performs training operations using a tiny version of the
    given dataset.

    :param dataset: DatasetDict - the full dataset
    :return: None
    """

    # Define global variables
    global train_iter
    global validation_iter
    global test_iter
    global _num_batches
    global _batch_size

    data = dataset

    print("Creating tiny dataset for test run...\n")

    # Create a tiny version of the dataset
    tiny_data = DatasetDict()
    tiny_data['train'] = data['train'].shuffle(seed=1).select(range(70000))
    tiny_data['validation'] = data['validation'].shuffle(seed=2).select(range(15000))
    tiny_data['test'] = data['test'].shuffle(seed=3).select(range(15000))

    # Convert dataset into Pandas DataFrame then create a data list for each split
    tiny_data.set_format('pandas')
    train_data = tiny_data['train'][:]
    test_data = tiny_data['test'][:]
    validation_data = tiny_data['validation'][:]

    # Calulate number of batches based on size of training dataset
    _num_batches = len(train_data) / _batch_size
    if isinstance(_num_batches, float):
        _num_batches = len(train_data) // _batch_size
    else:
        _num_batches = (len(train_data) // _batch_size) - 1

    # Initialize iterable lists for each split
    init_iterators(train_data, validation_data, test_data)

    # Get a basic english tokenizer to use for custom vocabulary
    tokenizer = get_tokenizer("basic_english")

    # Build the vocabulary for the model
    vocabulary = build_vocab(tokenizer, train_iter)

    # Define the vocab size based on the vocab object
    vocab_size = len(vocabulary)

    # Initialize the text and label pipelines for the dataloaders
    init_pipelines(vocabulary, tokenizer)

    # Initialize the DataLoaders for training
    init_dataloaders()

    # Define the model, loss criterion, optimizer, and scheduler
    model = TextClassificationModel(vocab_size, _embed_size, _num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)

    print("Preparing the model for training...\n")

    # Train the model
    train_model(model, scheduler, optimizer)

    return


def use_full_dataset(dataset):
    """
    Performs training operations using the full dataset.

    :param dataset: DatasetDict - the full dataset
    :return: None
    """

    # Define global variables
    global train_iter
    global validation_iter
    global test_iter
    global _num_batches
    global _batch_size

    data = dataset

    print("Preparing the dataset for training...\n")

    # Convert dataset into Pandas DataFrame then create a data list for each split
    data.set_format('pandas')
    train_data = data['train'][:]
    test_data = data['test'][:]
    validation_data = data['validation'][:]

    # Calulate number of batches based on size of training dataset
    _num_batches = len(train_data) / _batch_size
    if isinstance(_num_batches, float):
        _num_batches = len(train_data) // _batch_size
    else:
        _num_batches = (len(train_data) // _batch_size) - 1

    # Initialize iterable lists for each split
    init_iterators(train_data, validation_data, test_data)

    # Get a basic english tokenizer to use for custom vocabulary
    tokenizer = get_tokenizer("basic_english")

    # Build the vocabulary for the model
    vocabulary = build_vocab(tokenizer, train_iter)

    # Define the vocab size based on the vocab object
    vocab_size = len(vocabulary)

    # Initialize the text and label pipelines for the dataloaders
    init_pipelines(vocabulary, tokenizer)

    # Initialize the DataLoaders for training
    init_dataloaders()

    # Define the model, loss criterion, optimizer, and scheduler
    model = TextClassificationModel(vocab_size, _embed_size, _num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)

    print("Preparing the model for training...\n")

    # Train the model
    train_model(model, scheduler, optimizer)

    return


def main():
    """
    The main function of the script.

    :return: None
    """

    print("\nPreparing dataset for training...")

    data = prepare_dataset(source_fpath=source_filepath)

    input_valid = False

    # Prompt user for input related to generator mode
    while not input_valid:
        user_input = input("\nWant to perform a test run using a tiny dataset?  (Y/n)?")
        user_input = user_input.lower()

        if user_input == "y" or user_input == "n" or user_input == "yes" or user_input == "no":

            input_valid = True

            if user_input == "y" or user_input == "yes":
                print("\nRunning test run using a tiny version of the dataset...\n")
                use_tiny_dataset(data)
            else:
                print("\nProceeding with training using the full dataset...\n")
                use_full_dataset(data)
        else:
            print("\n  - Invalid Input - Please try again.")

    exit()


if __name__ == "__main__":
    main()