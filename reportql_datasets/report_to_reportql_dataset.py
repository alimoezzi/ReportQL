import os
import json
import pandas as pd
from torch.utils.data import Dataset
from transformers import T5Tokenizer
from utils import add_special_tokens
from config import configs
from utils import reportQL_to_json


class Report2ReportQLDataset(Dataset):
    def __init__(self, tokenizer, data_dir, schema_path, type_path, max_len=configs['max_seq_length'], **kwargs):
        super().__init__(**kwargs)
        self.path = os.path.join(data_dir, type_path + '.csv')
        with open(os.path.join(schema_path), mode='r') as sch:
            self.schema = json.load(sch)
        self.source_column = "report"
        self.target_column = "reportql"
        self.data = pd.read_csv(self.path)

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        for idx in range(len(self.data)):
            input_, target = self.data.loc[idx, self.source_column], self.data.loc[idx, self.target_column]

            organs = reportQL_to_json(target).keys()
            report_schema = [
                f' <t> {organ["name"]} ' + "{ " + ' '.join([field["name"] for field in organ["fields"]]) + " } </t>"
                for organ in self.schema["types"] if organ["name"] in organs
            ]

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [f"r: {input_.strip().lower()} s:{'{ ' + ''.join(report_schema) + ' }'}"], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


if __name__=='__main__':
    tokenizer = T5Tokenizer.from_pretrained(configs['tokenizer_name_or_path'])
    add_special_tokens(tokenizer)
    dataset = Report2ReportQLDataset(tokenizer, schema_path=configs['schema_path'], data_dir=configs['data_dir'], type_path='training')
    print(len(dataset))
    data = dataset[1]
    print(tokenizer.decode(data['source_ids']))
    print(tokenizer.decode(data['target_ids']))


def get_dataset(tokenizer, type_path, args, task='finetune'):
    if task=='finetune':
        return Report2ReportQLDataset(tokenizer=tokenizer, schema_path=configs['schema_path'], data_dir=configs['data_dir'], type_path=type_path,
                                      max_len=args.max_seq_length)
    else:
        raise NotImplementedError()
