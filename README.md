# Code AI Training Workflow

## Prerequisites
- Python 3.10+
- `pip install -r requirements.txt`
- Environment variable `GEMINI_API_KEY` (create `.env` then `source .env`)

## Train the base LoRA adapter
```
python scripts/train.py --config config/train.yaml
```
This produces `outputs/codellama-coder/` with tokenizer, adapter weights, and metadata.

## Run inference or ask Gemini for feedback
```
python src/run.py
```
(Feel free to edit `src/run.py` with your own prompts.)

## Iterative improvement with Gemini
1. Make sure you have a trained adapter and API key available.
2. Run the feedback loop to augment the dataset with Gemini-guided fixes:
   ```
   python scripts/feedback_loop.py \
     --run-dir outputs/codellama-coder \
     --dataset training/data/coding_qa.jsonl \
     --output training/data/coding_qa_augmented.jsonl \
     --max-samples 20
   ```
   - Gemini reviews each selected prompt/answer, supplies a score, and if the score is below the threshold it rewrites the answer using its own feedback.
   - The script writes an augmented dataset plus a `.feedback.log.jsonl` file containing the review details.
3. Re-run training using the augmented dataset:
   ```
   python scripts/train.py --config config/train.yaml
   ```
   (Before running, edit `config/train.yaml` so `dataset_path` points to
   `training/data/coding_qa_augmented.jsonl`, or create a copy of the config
   with that path.)

Repeat the loop (train ➝ feedback ➝ train) to continually adapt the model using Gemini's critiques.
