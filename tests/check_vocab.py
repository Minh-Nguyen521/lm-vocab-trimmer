import json

TRIMMED_MODEL = "model/gemma300-vi-trimmed-138"

with open(f"{TRIMMED_MODEL}/tokenizer.json") as f:
    tok = json.load(f)

vocab = tok["model"]["vocab"]

vi_chars = (
    list("abcdefghijklmnopqrstuvwxyz")
    + list("àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ")
    + [str(i) for i in range(10)]
)

missing = [repr(ch) for ch in vi_chars if not any(t.strip("▁") == ch for t in vocab)]

print(f"Vocab size : {len(vocab):,}")
print(f"Checked    : {len(vi_chars)} chars (a-z, A-Z, Vietnamese diacritics, 0-9)")
print(f"Missing    : {missing if missing else 'None - all present'}")
