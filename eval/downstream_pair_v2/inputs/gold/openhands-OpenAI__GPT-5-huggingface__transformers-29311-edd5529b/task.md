<uploaded_files>
/workspace/huggingface__transformers__b27aa206ddf3fe66b36db587603141b3d0379a82
</uploaded_files>

I've uploaded a python code repository in the directory huggingface__transformers__b27aa206ddf3fe66b36db587603141b3d0379a82. Consider the following issue description:

<issue_description>
`skip_special_tokens` for `Wav2Vec2CTCTokenizer` does not work expectedly.
### System Info

- `transformers` version: 4.37.2
- Platform: Linux-5.15.0-1042-nvidia-x86_64-with-glibc2.35
- Python version: 3.10.13
- Huggingface_hub version: 0.20.1
- Safetensors version: 0.4.2
- Accelerate version: 0.26.1
- Accelerate config:    not found
- PyTorch version (GPU?): 2.1.2 (True)
- Tensorflow version (GPU?): not installed (NA)
- Flax version (CPU?/GPU?/TPU?): not installed (NA)
- Jax version: not installed
- JaxLib version: not installed
- Using GPU in script?: Yes
- Using distributed or parallel set-up in script?: DDP

### Who can help?

@sanchit-gandhi @ArthurZucker 

### Information

- [ ] The official example scripts
- [X] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [X] My own task or dataset (give details below)

### Reproduction

```python
processor = Wav2Vec2Processor.from_pretrained(model_args.model_name_or_path)
model = Wav2Vec2ConformerForCTC.from_pretrained(
    model_args.model_name_or_path, ctc_loss_reduction="mean", pad_token_id=processor.tokenizer.pad_token_id
)

metric_wer = evaluate.load("wer")
metric_cer = evaluate.load("cer")

def _compute_metrics(pred):
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    # ["<s>": 1, "</s>": 2, "<unk>": 3, "<pad>": 0]
    preds = processor.batch_decode(pred.predictions, skip_special_tokens=True)
    labels = processor.batch_decode(pred.label_ids, skip_special_tokens=True, group_tokens=False)

    preds = [postprocess(text) if len(text) != 0 else "-" for text in preds]
    labels = [postprocess(sentence) for sentence in labels]

    preds = [re.sub(r"\s+", "", text) for text in preds]
    labels = [re.sub(r"\s+", "", text) for text in labels]

    wer = 100 * metric_wer.compute(predictions=preds, references=labels)
    cer = 100 * metric_cer.compute(predictions=preds, references=labels)

    return {"wer": wer, "cer": cer}

def _preprocess_logits_for_metrics(logits, labels=None):
    return torch.argmax(logits, dim=-1)

trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=processor,
    compute_metrics=_compute_metrics,
    preprocess_logits_for_metrics=_preprocess_logits_for_metrics,
)
```

### Expected behavior

I want to train ASR model and this issue came out when I evaluate during training.
`preds = processor.batch_decode(pred.predictions, skip_special_tokens=True)`'s results should remove all special tokens (`<s>`, `<\s>`, `<unk>`, `<pad>`). However, in my experiment, `<unk>` is not removed.

So, I look at the code for `Wav2Vec2CTCTokenizer._decode` (transformers / models / wav2vec2 / tokenization_wav2vec2.py).
```python
 def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        group_tokens: bool = True,
        spaces_between_special_tokens: bool = False,
        output_word_offsets: Optional[bool] = False,
        output_char_offsets: Optional[bool] = False,
    ) -> str:
        """
        special _decode function is needed for Wav2Vec2Tokenizer because added tokens should be treated exactly the
        same as tokens of the base vocabulary and therefore the function `convert_tokens_to_string` has to be called on
        the whole token list and not individually on added tokens
        """
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        result = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            result.append(token)
```
This code removes the special tokens in the `filtered_tokens` by `self.all_special_ids`, but when I print the `filtered_tokens`, the outcome looks like `['|', '|', 'token1', 'token2', 'token3', '|', '|', 'token4', '|', '|', '|', 'token5', 'token6', '|', '|', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>']`. 

Since `self.all_special_ids`'s elements are integers for special tokens, `if skip_special_tokens and token in self.all_special_ids:` statement does not work expectedly.

Shouldn’t it be `if skip_special_tokens and token in self.all_special_tokens:`?

</issue_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Also the development Python environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.
Your task is to make the minimal changes to non-test files in the /workspace/huggingface__transformers__b27aa206ddf3fe66b36db587603141b3d0379a82 directory to ensure the <issue_description> is satisfied.

Follow these phases to resolve the issue:

Phase 1. READING: read the problem and reword it in clearer terms
   1.1 If there are code or config snippets. Express in words any best practices or conventions in them.
   1.2 Hightlight message errors, method names, variables, file names, stack traces, and technical details.
   1.3 Explain the problem in clear terms.
   1.4 Enumerate the steps to reproduce the problem.
   1.5 Hightlight any best practices to take into account when testing and fixing the issue

Phase 2. RUNNING: install and run the tests on the repository
   2.1 Follow the readme
   2.2 Install the environment and anything needed
   2.2 Iterate and figure out how to run the tests

Phase 3. EXPLORATION: find the files that are related to the problem and possible solutions
   3.1 Use `grep` to search for relevant methods, classes, keywords and error messages.
   3.2 Identify all files related to the problem statement.
   3.3 Propose the methods and files to fix the issue and explain why.
   3.4 From the possible file locations, select the most likely location to fix the issue.

Phase 4. TEST CREATION: before implementing any fix, create a script to reproduce and verify the issue.
   4.1 Look at existing test files in the repository to understand the test format/structure.
   4.2 Create a minimal reproduction script that reproduces the located issue.
   4.3 Run the reproduction script to confirm you are reproducing the issue.
   4.4 Adjust the reproduction script as necessary.

Phase 5. FIX ANALYSIS: state clearly the problem and how to fix it
   5.1 State clearly what the problem is.
   5.2 State clearly where the problem is located.
   5.3 State clearly how the test reproduces the issue.
   5.4 State clearly the best practices to take into account in the fix.
   5.5 State clearly how to fix the problem.

Phase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen solution.
   6.1 Make minimal, focused changes to fix the issue.

Phase 7. VERIFICATION: Test your implementation thoroughly.
   7.1 Run your reproduction script to verify the fix works.
   7.2 Add edge cases to your test script to ensure comprehensive coverage.
   7.3 Run existing tests related to the modified code to ensure you haven't broken anything.

8. FINAL REVIEW: Carefully re-read the problem description and compare your changes with the base commit b27aa206ddf3fe66b36db587603141b3d0379a82.
   8.1 Ensure you've fully addressed all requirements.
   8.2 Run any tests in the repository related to:
     8.2.1 The issue you are fixing
     8.2.2 The files you modified
     8.2.3 The functions you changed
   8.3 If any tests fail, revise your implementation until all tests pass

Be thorough in your exploration, testing, and reasoning. It's fine if your thinking process is lengthy - quality and completeness are more important than brevity.