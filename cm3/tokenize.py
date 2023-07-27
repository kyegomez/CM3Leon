import multiprocessing 
import argparse
from itertools import chain
from datasets import load_dataset
from cm3.model import CM3LEONTokenizer

class CFG:
    SEED: int = 42
    SEQ_LEN: int = 8192
    NUM_CPU: int = multiprocessing.cpu_count()
    HF_ACCOUNT_REPO: str = "YOUR HUGGINGFACE API KEY"
    DATASET_NAME: str = "HuggingFaceM4/VQAv2"


#perhaps will need finetuning
def built_dataset(args):
    # tokenizer = AutoTokenizer.from_pretrained(CFG.TOKENIZER)
    tokenizer = CM3LEONTokenizer.tokenize
    
    train_dataset = load_dataset(CFG.DATASET_NAME, split="train", streaming=True)

    def tokenize_function(example):
        return tokenizer([t + tokenizer.eos_token for t in example["text"]])
    
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=CFG.NUM_CPU,
        remove_columns=["text"],
    )

    block_size = CFG.SEQ_LEN


    #main data processing functin that will concatenate all texts from our dataset
    def group_texts(examples):
        #concatenate all texts
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        #drop the small remainder we could add padding if the model supported it instead of this drop customize
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        
        #split by chunks of max length
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        return result
    

    train_tokenized_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=CFG.NUM_PROC,
    )

    train_tokenized_dataset.push_to_hub(CFG.HF_ACCOUNT_REPO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and push dataset to Hugging Face Hub")
    parser.add_argument("--seed", type=int, default=CFG.SEED, help="Random seed")
    parser.add_argument("--seq_len", type=int, default=CFG.SEQ_LEN, help="Sequence length for processing")
    parser.add_argument("--hf_account", type=str, default=CFG.HF_ACCOUNT_REPO, help="Hugging Face account name and repo")
    parser.add_argument("--tokenizer", type=str, default=CFG.TOKENIZER, help="Tokenizer model to use")
    parser.add_argument("--dataset_name", type=str, default=CFG.DATASET_NAME, help="Name of the dataset to process")
    args = parser.parse_args()
    built_dataset(args)
