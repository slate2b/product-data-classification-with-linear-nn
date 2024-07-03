
import torch
from datasets import DatasetDict
from preprocessor import prepare_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch import nn
import time
from torchtext.data.functional import to_map_style_dataset

source_filepath = './source_data/amazon-products-text-and-label_ids.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
_epochs = 15
_learning_rate = 5
_batch_size = 64
_num_labels = 20
_embed_size = 1024


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

print("\nPreparing dataset for training...\n")

data = prepare_dataset(source_fpath=source_filepath)

tiny_data = DatasetDict()
tiny_data['train'] = data['train'].shuffle(seed=1).select(range(70000))
tiny_data['validation'] = data['validation'].shuffle(seed=1).select(range(15000))
tiny_data['test'] = data['test'].shuffle(seed=1).select(range(15000))

tokenizer = get_tokenizer("basic_english")

# data.set_format('pandas')
# train_data_df = data['train'][:]
# test_data_df = data['test'][:]
# validation_data_df = data['validation'][:]
tiny_data.set_format('pandas')
train_data_df = tiny_data['train'][:]
test_data_df = tiny_data['test'][:]
validation_data_df = tiny_data['validation'][:]

def build_iter(d_frame):
    """
    Builds a list of tuples containing the labels and text from a Pandas DataFrame.

    :param d_frame: Pandas DataFrame - A dataframe version of the dataset split
    :return: iter_list - list of tuples containing the labels and text from the given df
    """

    iter_list = []

    for i in range(len(d_frame.index)):
        label = train_data_df['label'].iloc[i]
        text = train_data_df['text'].iloc[i]
        iter_tuple = (label, text)
        iter_list.append(iter_tuple)

    return iter_list


# Create iterable lists for each dataset split
train_iter = build_iter(train_data_df)
test_iter = build_iter(test_data_df)
validation_iter = build_iter(validation_data_df)

# Build the vocabulary using the tokenizer defined in global variables
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Create pipelines for text processing
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x)


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)

    return label_list.to(device), text_list.to(device), offsets.to(device)


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.do1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(embed_dim, embed_dim // 2)
        self.do2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(embed_dim // 2, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)

        f1 = self.fc1(embedded)
        d1 = self.do1(f1)
        f2 = self.fc2(d1)
        d2 = self.do2(f2)
        output = self.fc3(d2)
        return output


# Define the vocab size based on the vocab object
vocab_size = len(vocab)

# Define the model and the model, loss criterion, optimizer, and scheduler
model = TextClassificationModel(vocab_size, _embed_size, _num_labels).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=_learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
total_accu = None

# Convert the dataset splits to map style datasets
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
validation_dataset = to_map_style_dataset(validation_iter)

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


def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label) * 0.0001
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return [total_acc / total_count, loss]


for epoch in range(1, _epochs + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    eval_results = evaluate(valid_dataloader)
    accu_val = eval_results[0]
    loss_val = eval_results[1]
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print("-" * 84)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} | valid loss {:8.3f}".format(
            epoch, time.time() - epoch_start_time, accu_val, loss_val
        )
    )
    print("-" * 84)

print("Checking the results of test dataset.")
accu_test = evaluate(test_dataloader)
print("test accuracy {:8.3f}".format(accu_test))
