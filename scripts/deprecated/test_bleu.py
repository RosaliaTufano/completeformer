import datasets

predictions = [
    # ["hello", "there", "general", "kenobi"],
    ["foo", "bar", "foobar", "barfoo"],
    ["foo", "bar", "foobar", "hello", "there"]
]
references = [
    # [["hello", "there", "general", "kenobi"]],
    # [["hello", "there", "kenobi", "you", "are", "a" "general"]],
    [["hello", "there", "foo", "bar", "foobar", "barfoo"]],
    [["hello", "there", "foo", "bar", "foobar", "hello", "there"]]
]
bleu = datasets.load_metric("bleu")
results = bleu.compute(predictions=predictions, references=references)["bleu"]
print(results)

predictions = [
    # ["hello", "there", "general", "kenobi"],
    # ["foo", "bar", "foobar", "barfoo"],
    ["foo", "bar", "foobar", "hello", "there"]
]
references = [
    # [["hello", "there", "general", "kenobi"]],
    # [["hello", "there", "kenobi", "you", "are", "a" "general"]],
    # [["hello", "there", "foo", "bar", "foobar", "barfoo"]],
    [["hello", "there", "foo", "bar", "foobar", "hello", "there"]]
]
# bleu = datasets.load_metric("bleu")
results1 = bleu.compute(predictions=predictions, references=references)["bleu"]
# print(results)

predictions = [
    # ["hello", "there", "general", "kenobi"],
    ["foo", "bar", "foobar", "barfoo"],
    # ["foo", "bar", "foobar", "hello", "there"]
]
references = [
    # [["hello", "there", "general", "kenobi"]],
    # [["hello", "there", "kenobi", "you", "are", "a" "general"]],
    [["hello", "there", "foo", "bar", "foobar", "barfoo"]],
    # [["hello", "there", "foo", "bar", "foobar", "hello", "there"]]
]
# bleu = datasets.load_metric("bleu")
results2 = bleu.compute(predictions=predictions, references=references)["bleu"]
print((results1 + results2) / 2)